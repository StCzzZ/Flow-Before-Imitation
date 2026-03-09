import os
import time
import yaml
import numpy as np
from skimage import measure
import trimesh
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pysdf import SDF
from diso import DiffDMC

from networks.pointnet.pointnet import PointNetfeat
from networks.pointnet.pointnet_plus import PointNetPlus
from networks.layer import EquivariantLayer
from networks.transformer import TransformerFusion
from networks.corr_module import CorrFusion, CorrFlowFusion
from networks.hand_tracking_pipeline import HandPoseTrackingModule

from components.unet3d import UNet3D
from networks.decoder import WNFdecoder
from components.manopth.manopth.manolayer_old import ManoLayer
from components.manopth.manopth.inverse_kinematics import ik_solver_mano

from networks.hand_estimator import HandTracker, HandTracker_v2, MLPDecoder

from common.eval_utils import compute_iou
from common.coordinate_utils import normalize_3d_coordinate, coordinate2index
from common.grid_utils import make_3d_grid
from common.pointcloud_utils import norm_pointcloud, norm_pointcloud_batch, rotate_hand_points_tensor, chamfer_distance,\
    denorm_pointcloud_batch, inverse_rotate_hand_points_tensor
from common.io_utils import write_ply

from torch_scatter import scatter_mean, scatter_max
### Load the checkpoint with this function
from pytorch_lightning.utilities.cloud_io import load as pl_load

torch.set_default_dtype(torch.float32)


class BaseTrackingPipeline(pl.LightningModule):
    def __init__(self):
        super().__init__()
        ### Training parameters
        self.lr = None
        self.optim = None

        ### Pointnet
        self.pc_feature_encoder = None
        
        ### Transformer or Corr Fusion
        self.transformerfuser = None
        self.corr_fuser = None
        
        ### Grid feature Generator
        self.grid_dim = None
        self.reso_grid = None
        self.padding = None
        
        ### Grid feature Sampler
        self.sample_mode = None
        self.unet3d = None

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = None
        
        ### Visualization param
        self.vis_every = None
        self.nx = None
        self.box_size = None
        self.vis_point_every_split = None
        self.print_vis_time = None
        self.vis_flow = None
        
        
        ### Hand pose estimator
        self.hand_tracker = None
        
        self.loss_items = None
    
    def load_model_only(self, chkpt_path):
        ckpt = pl_load(chkpt_path, map_location=self.device)
        self.load_state_dict(ckpt['state_dict'])
    
    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.grid_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x D x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.grid_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x D x reso^3)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid
    
    def sample_grid_feature(self, sample_p, c):
        p_nor = normalize_3d_coordinate(sample_p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        
        return c.permute(0, 2, 1)

    def forward_manolayer(self, hand_pose, pc_for_norm):
        B = hand_pose.size(0)
        mano_hand_trans, mano_hand_rot = hand_pose[:, :3], hand_pose[:, 3:]
        mano_info = self.hand_tracker.manolayer(mano_hand_rot)
        
        mano_verts, mano_jtrs, mano_transf, _ = mano_info
        # mano_verts += mano_hand_trans
        mano_verts = rotate_hand_points_tensor(mano_verts, mano_hand_trans, self.device)
        mano_verts = norm_pointcloud_batch(mano_verts, pc_for_norm)
        mano_faces = self.hand_tracker.manolayer.th_faces
        
        mano_jtrs = rotate_hand_points_tensor(mano_jtrs, mano_hand_trans, self.device)
        mano_jtrs = norm_pointcloud_batch(mano_jtrs, pc_for_norm)
        
        mano_trans_cat = torch.cat(
            [
                torch.cat([torch.zeros(B, 3, 3).to(self.device), mano_hand_trans.view(B, 3, -1)], dim=2),
                torch.zeros(B, 1, 4).to(self.device),
            ],
            dim=1,
        ).unsqueeze(1)
        mano_transf = mano_transf + mano_trans_cat
        
        mano_info = {'mano_verts': mano_verts, 
                     'mano_faces': mano_faces, 
                     'mano_jtrs': mano_jtrs,
                     'mano_transf': mano_transf}
        
        return mano_info
    
    def forward(self, pc_1, pc_2, sample_p):
        ### Predict wnf for sample_p
        out_wnf = self.forward_obj(pc_1, pc_2, sample_p)
        
        ### Predict hand pose
        if self.hand_tracker is not None:
            mano_pred = self.hand_tracker(pc_1, pc_2)
            return out_wnf, mano_pred
        
        return out_wnf, None

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        occ_pred, mano_pred = self.forward(pc_1, pc_2, sample_p)
        
        if "sdf" in self.loss_items.keys():
            loss = self.loss_items['sdf'] * F.l1_loss(occ_pred, occ)
            self.log("occ_loss", loss)
        
        if (mano_pred is not None) and ("hand" in self.loss_items.keys()):
            loss_hand = self.loss_items['hand'] * F.mse_loss(mano_pred, mano)
            self.log("hand_loss", loss_hand)
            
            loss += loss_hand
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        occ_pred, *_ = self.forward(pc_1, pc_2, sample_p)
        iou = compute_iou(occ1=occ.detach().cpu(), occ2=occ_pred.detach().cpu(), threshold=0.5)[0]
        print("Validation metric (iou):", iou)
        self.log("iou", iou)
        
        return iou
    
    def extract_mesh(self, extractor, occ_grid, padding):
        verts_diso, faces_diso = extractor(occ_grid)
        verts_diso -= 0.5 * torch.ones(3).to(self.device)
        verts_diso *= (1 + padding)

        return verts_diso, faces_diso
    
    def vis_obj(self, pc_1, pc_2, sample_p_grid_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            occ_pred_list = []
            for sample_p in sample_p_grid_split:
                sample_p = sample_p.to(self.device)
                occ_pred = self.forward_obj(pc_1, pc_2, sample_p.unsqueeze(0))
                occ_pred_list.append(occ_pred)
        
        occ_pred = torch.cat(occ_pred_list, dim=1).to(self.device)
        
        occ_grid = occ_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, occ_grid, self.padding, self.padding)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return occ_grid, vertices, faces
    
    def vis_hand(self, pc_1, pc_2, pc_for_norm):
        ### Used on callback for visulization, or geo fit
        mano_pred = self.hand_tracker(pc_1, pc_2)
        mano_info = self.forward_manolayer(mano_pred, pc_for_norm)
        mano_verts, mano_faces, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']

        return mano_pred, mano_verts, mano_faces
    
    def vis_step(self, batch, batch_idx):
        ### Used on callback for visulization, or geo fit
        start_vis_time = time.time()
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, occ, mano, pc_for_norm = pc_1.float(), pc_2.float(), occ.float(), mano.float(), pc_for_norm.float()
        
        B, N, C = pc_2.size()
        sample_p_grid = self.box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_split = torch.split(sample_p_grid, self.vis_point_every_split)
        
        with torch.no_grad():
            occ_grid_list, verts_obj_list, face_obj_list = [], [], []
            for b in range(B):
                pc_1_batch, pc_2_batch = pc_1[b:b+1], pc_2[b:b+1]
                occ_grid, verts_obj, face_obj = self.vis_obj(pc_1_batch, pc_2_batch, sample_p_grid_split)
                verts_obj, face_obj = np.array(verts_obj, dtype=np.float32), np.array(face_obj, dtype=np.int32)

                occ_grid_list.append(occ_grid)
                verts_obj_list.append(verts_obj)
                face_obj_list.append(face_obj)
            
            if self.print_vis_time:
                print("Vis time: ", time.time() - start_vis_time)

            if self.hand_tracker is not None:
                mano_pred, verts_hand, face_hand = self.vis_hand(pc_1_batch, pc_2_batch, pc_for_norm)
                
                ### Test if hand vis is correct
                # mano_pred = None
                # mano_info = self.forward_manolayer(mano_pred, pc_for_norm)
                # verts_hand, face_hand, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']
                
                verts_hand, face_hand = verts_hand.detach().cpu().numpy(), face_hand.cpu().numpy()
                
            else:
                verts_hand, face_hand = None, None
                    
        return name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand, face_hand

    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
    
    
class WNFTrackingPipeline(BaseTrackingPipeline):
    def __init__(self,
                 training_param,
                 pointnet_param,
                 transformer_param,
                 grid_param,
                 unet3d_param,
                 decoder_param,
                 vis_param,
                 loss_items,
                 hand_decoder_param=None,
                 ):
        super().__init__()
        
        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"

        ### Pointnet
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
        ### Transformer Fusion
        self.transformerfuser = TransformerFusion(**transformer_param)
        
        ### Grid feature Generator
        self.grid_dim = grid_param['grid_dim']
        self.reso_grid = grid_param['reso_grid']
        self.padding = grid_param['padding']
        
        ### Grid feature Sampler
        self.sample_mode = grid_param['sample_mode']
        self.unet3d = UNet3D(**unet3d_param)
        

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = WNFdecoder(**grid_param, **decoder_param)
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.nx = vis_param['nx']
        self.box_size = 1 + self.padding
        self.vis_point_every_split = vis_param['vis_point_every_split']
        self.print_vis_time = vis_param['print_time']
        
        ### Loss items
        self.loss_items = loss_items
        
        ### Hand pose estimator
        if hand_decoder_param is not None:
            if "transformer_param" in hand_decoder_param:
                self.hand_tracker = HandTracker_v2(**hand_decoder_param)
            else:
                self.hand_tracker = HandTracker(**hand_decoder_param)
        else:
            self.hand_tracker = None
    
    def forward_obj(self, pc_1, pc_2, sample_p):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Transformer Fusion feature from two pc
        pc_feat_fused = self.transformerfuser(pc_2_feat, pc_2, pc_1_feat, pc_1)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_fused_grid = self.generate_grid_features(pc_2, pc_feat_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_feat = self.sample_grid_feature(sample_p, pc_feat_fused_grid)
        
        ### Decode wnf from sample_p_feat
        out_wnf = self.decoder(sample_p, sample_p_feat)
        
        return out_wnf
    
    
class WNFTrackingCorrFlowPipeline(BaseTrackingPipeline):
    def __init__(self,
                 training_param,
                 pointnet_param,
                 Corr_param,
                 grid_param,
                 unet3d_param,
                 decoder_param,
                 vis_param,
                 loss_items, 
                 hand_decoder_param=None,
                 ):
        super().__init__()
        
        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"

        ### Pointnet
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
        ### Correspondance Fusion
        self.corr_fuser = CorrFlowFusion(**Corr_param)
        
        ### Grid feature Generator
        self.grid_dim = grid_param['grid_dim']
        self.reso_grid = grid_param['reso_grid']
        self.padding = grid_param['padding']
        
        ### Grid feature Sampler
        self.sample_mode = grid_param['sample_mode']
        
        self.unet3d = UNet3D(**unet3d_param)
        

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = WNFdecoder(**grid_param, **decoder_param)
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.nx = vis_param['nx']
        self.box_size = 1 + self.padding
        self.vis_point_every_split = vis_param['vis_point_every_split']
        self.print_vis_time = vis_param['print_time']
        self.vis_flow = vis_param['vis_flow']
        
        ### Loss items
        self.loss_items = loss_items
        
        ### Hand pose estimator
        if hand_decoder_param is not None:
            if "transformer_param" in hand_decoder_param:
                self.hand_tracker = HandTracker_v2(**hand_decoder_param)
            else:
                self.hand_tracker = HandTracker(**hand_decoder_param)
        else:
            self.hand_tracker = None
    
    def forward_obj(self, pc_1, pc_2, sample_p):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_fused_grid = self.generate_grid_features(pc_2, pc_feat_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_feat = self.sample_grid_feature(sample_p, pc_feat_fused_grid)
        
        ### Decode wnf from sample_p_feat
        out_wnf = self.decoder(sample_p, sample_p_feat)
        
        return out_wnf, pred_corr, pred_flow
    
    def forward(self, pc_1, pc_2, sample_p):
        ### Predict wnf for sample_p
        out_wnf, pred_corr, pred_flow = self.forward_obj(pc_1, pc_2, sample_p)
        
        ### Predict hand pose
        if self.hand_tracker is not None:
            mano_pred = self.hand_tracker(pc_1, pc_2)
            return out_wnf, mano_pred, pred_corr, pred_flow
        
        return out_wnf, None, pred_corr, pred_flow
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        B, N, C = pc_2.size()
        ### Total loss
        loss = 0
        
        occ_pred, mano_pred, pred_corr, pred_flow = self.forward(pc_1, pc_2, sample_p)
        
        ### SDF prediction loss
        if "sdf" in self.loss_items.keys():
            loss_occ = F.l1_loss(occ_pred, occ)
            loss += loss_occ
            self.log("occ_loss", loss_occ)
        
        ### Correspondance self-supervised loss
        if "corr" in self.loss_items.keys():
            loss_corr = 1 - torch.mean(torch.max(pred_corr, dim=2)[0])
            loss += loss_corr
            self.log("corr_loss", loss_corr)
        
        ### Flow self-supervised loss
        if "flow" in self.loss_items.keys():
            # pc_2_corr = torch.einsum('bji,bik->bjk', pred_corr, pc_2.permute(0, 2, 1))
            # pc_1_corr = torch.einsum('bji,bik->bjk', pred_corr_back, pc_1.permute(0, 2, 1))
            # loss_flow = chamfer_distance((pc_1 + pred_flow).permute(0, 2, 1), pc_2_corr)
            loss_flow = chamfer_distance((pc_1.permute(0, 2, 1) + pred_flow).permute(0, 2, 1), pc_2)
            loss += loss_flow
            
            ### Backward flow prediction
            pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
            pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
            ### Correspondance Fusion feature from two pc
            pred_corr_back, pred_flow_back, pc_feat_fused_back = self.corr_fuser(pc_2, pc_1, pc_2_feat, pc_1_feat)
        
            # loss_flow_back = chamfer_distance((pc_2 + pred_flow_back).permute(0, 2, 1), pc_1_corr)
            loss_flow_back = chamfer_distance((pc_2.permute(0, 2, 1) + pred_flow_back).permute(0, 2, 1), pc_1)
            # loss += loss_flow_back
            
            self.log("flow_loss", loss_flow)
            
            
        
        if (mano_pred is not None) and ("hand" in self.loss_items.keys()):
            loss_hand = self.loss_items['hand'] * F.mse_loss(mano_pred, mano)
            
            mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
            hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
            
            mano_info_pred = self.forward_manolayer(mano_pred, pc_for_norm)
            hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']
            
            loss_hand_verts = self.loss_items['hand'] * F.mse_loss(hand_verts_pred, hand_verts_gt)
            loss_hand_verts += self.loss_items['hand'] * F.mse_loss(hand_jtrs_pred, hand_jtrs_gt)
            
            loss_hand += loss_hand_verts
            
            self.log("hand_loss", loss_hand)
            
            loss += loss_hand
    
        return loss
    
    def vis_obj(self, pc_1, pc_2, sample_p_grid_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            occ_pred_list = []
            for sample_p in sample_p_grid_split:
                sample_p = sample_p.to(self.device)
                occ_pred, pred_corr, pred_flow = self.forward_obj(pc_1, pc_2, sample_p.unsqueeze(0))
                occ_pred_list.append(occ_pred)
        
        occ_pred = torch.cat(occ_pred_list, dim=1).to(self.device)

        occ_grid = occ_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, occ_grid, self.padding)
        print(verts_diso.size(), faces_diso.size())
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return occ_grid, vertices, faces
    
    
    def vis_corr_flow(self, batch, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2 = pc_1.float(), pc_2.float()
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        return pc_1.permute(0, 2, 1) + pred_flow
    


class WNFTrackingHandSDFPipeline(BaseTrackingPipeline):
    def __init__(self,
                 assets_dir,
                 training_param,
                 pointnet_param,
                 Corr_param,
                 grid_param,
                 unet3d_param,
                 decoder_param,
                 vis_param,
                 loss_items, 
                 hand_decoder_param=None,
                 ):
        super().__init__()
        
        ### Closed Mano hand faces
        self.mano_close_face = np.load(os.path.join(assets_dir, "closed_fmano.npy")).astype(np.int32)
        
        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"

        ### Pointnet
        self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        
        ### Correspondance Fusion
        self.corr_fuser = CorrFlowFusion(**Corr_param)
        
        ### Grid feature Generator
        self.grid_dim = grid_param['grid_dim']
        self.reso_grid = grid_param['reso_grid']
        self.padding = grid_param['padding']
        
        ### Grid feature Sampler
        self.sample_mode = grid_param['sample_mode']
        
        self.unet3d = UNet3D(**unet3d_param)
        

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = WNFdecoder(**grid_param, **decoder_param)
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.nx = vis_param['nx']
        self.box_size = 1 + self.padding
        self.vis_point_every_split = vis_param['vis_point_every_split']
        self.print_vis_time = vis_param['print_time']
        self.vis_flow = vis_param['vis_flow']
        
        ### Loss items
        self.loss_items = loss_items
        
        ### Hand pose estimator
        if hand_decoder_param is not None:
            self.num_kp = 22
            self.dense_soft_factor = 1.0
            
            self.pc_hand_feat_encoder = PointNetPlus(in_channel=3)
            
            self.hand_f_fuser = TransformerFusion(**hand_decoder_param['transformer_param'])
            
            self.cls_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
            # self.cls_2 = EquivariantLayer(64, self.num_classes, activation=None, normalization=None)

            self.kp_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
            self.kp_2 = EquivariantLayer(64, 3 * self.num_kp, activation=None, normalization=None)
            
            self.use_attention = True
            self.kp_att_1 = EquivariantLayer(128, 64, activation='relu', normalization='batch')
            self.kp_att_2 = EquivariantLayer(64, self.num_kp, activation=None, normalization=None)
                        
            self.manolayer = ManoLayer(**hand_decoder_param['manolayer_param'])
            
            # self.feat_mlp = MLPDecoder(**hand_decoder_param['feat_mlp_param'])
            self.hand_sdf_decoder = WNFdecoder(**grid_param, **hand_decoder_param['decoder_param'])

            ### Hand SDF Grid param
            self.hand_padding = hand_decoder_param['grid_param']['padding']
            self.hand_box_size = 1 + self.hand_padding
        else:
            self.hand_sdf_decoder = None
            
            
    def forward_manolayer(self, hand_pose, pc_for_norm):
        B, _ = hand_pose.size()
        mano_hand_trans, mano_hand_rot = hand_pose[:, :3], hand_pose[:, 3:]
        mano_info = self.manolayer(mano_hand_rot)
        
        mano_verts, mano_jtrs, mano_transf, _ = mano_info
        
        mano_verts = rotate_hand_points_tensor(mano_verts, mano_hand_trans, self.device)
        mano_verts = norm_pointcloud_batch(mano_verts, pc_for_norm)
        mano_faces = self.manolayer.th_faces
        
        mano_jtrs = rotate_hand_points_tensor(mano_jtrs, mano_hand_trans, self.device)
        mano_jtrs = norm_pointcloud_batch(mano_jtrs, pc_for_norm)
        
        mano_trans_cat = torch.cat(
            [
                torch.cat([torch.zeros(B, 3, 3).to(self.device), mano_hand_trans.view(B, 3, -1)], dim=2),
                torch.zeros(B, 1, 4).to(self.device),
            ],
            dim=1,
        ).unsqueeze(1)

        mano_transf = mano_transf + mano_trans_cat
        
        mano_info = {'mano_verts': mano_verts, 
                     'mano_faces': mano_faces, 
                     'mano_jtrs': mano_jtrs,
                     'mano_transf': mano_transf}
        
        return mano_info
    
    
    def forward_obj(self, pc_1, pc_2, sample_p):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_fused_grid = self.generate_grid_features(pc_2, pc_feat_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_feat = self.sample_grid_feature(sample_p, pc_feat_fused_grid)
        
        ### Decode wnf from sample_p_feat
        out_wnf = self.decoder(sample_p, sample_p_feat)
        
        return out_wnf, pc_feat_fused_grid, pred_flow
    
    def forward_hand_jtrs(self, pc_1, pc_2):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        feat = None
        pts = pc_2
        
        pc_1_feat = self.pc_hand_feat_encoder(pc_1, feat).contiguous()  # (bs, 128, n)
        pc_2_feat = self.pc_hand_feat_encoder(pc_2, feat).contiguous()  # (bs, 128, n)
        
        ### Transformer Fusion feature from two pc
        dense_pts_feat = self.hand_f_fuser(pc_2_feat, pc_2, pc_1_feat, pc_1).transpose(1, 2).contiguous()  # (bs, 128, n)
        
        # dense_pts_feat = self.pc_hand_feat_encoder(pts, feat).contiguous()  # (bs, 128, n)
        
        # global_pts_feat = torch.max(dense_pts_feat, dim=-1, keepdim=True)[0]  # (bs, 128, 1)
        # dense_cls_score = self.cls_2(self.cls_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, K)

        dense_kp_offset = self.kp_2(self.kp_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, M*3)
        dense_kp_offset = dense_kp_offset.reshape(B, N, -1, 3)  # (bs, n, M, 3)

        dense_kp_coords = pts.reshape(B, N, 1, 3).expand(B, N, self.num_kp,
                                                          3) + dense_kp_offset  # (bs, n, M, 3)
        if self.use_attention:
            dense_kp_score = self.kp_att_2(self.kp_att_1(dense_pts_feat))\
                .transpose(1, 2).contiguous().reshape(B, N, -1, 1)  # (bs, n, M, 1)
            dense_kp_prob = F.softmax(dense_kp_score * self.dense_soft_factor, dim=1)  # (bs, n, M, 1)
        else:
            dense_kp_prob = torch.ones_like(dense_kp_coords) / N
        
        pred_kp = torch.sum(dense_kp_prob * dense_kp_coords, dim=1, keepdim=True)  # (bs, 1, M, 3)
        pred_kp = pred_kp.squeeze(1)  # (bs, M, 3)

        mano_hand_trans = pred_kp[:, 0, :].squeeze(1)

        hand_info = {'jtrs': pred_kp, 
                     'hand_trans': mano_hand_trans}
        
        return hand_info
    
    def forward_hand_sdf(self, hand_info, sample_p_hand, pc_for_norm):
        B, N_s, _ = sample_p_hand.size()
        
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        ### Ik pose
        hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full[:, 1:, :], pc_for_norm)
        hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred, self.device)
        
        mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
        mano_ik_pose_pred = mano_ik_pred_info['pose']
        
        hand_info['pose'] = mano_ik_pose_pred
        
        ### SDF prediction
        # feat_pose = self.feat_mlp(mano_ik_pose_pred)
        feat_pose = mano_ik_pose_pred[:, :32]
        feat_p_sdf = feat_pose.unsqueeze(1).repeat(1, N_s, 1)
        
        hand_sdf = self.hand_sdf_decoder(sample_p_hand, feat_p_sdf)
        
        hand_info['sdf'] = hand_sdf
        return hand_info

    
    def generate_sample_p_hand(self, mano, pc_for_norm, N_s):
        B = mano.size(0)
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        sample_p_hand_random_list = []
        for b in range(B):
            sample_p_hand_b = np.array(np.array([-self.hand_box_size/2, -self.hand_box_size/2, -self.hand_box_size/2]) + np.random.rand(N_s - hand_verts_gt.size(1), 3)*self.hand_box_size, dtype=np.float32)
            sample_p_hand_random_list.append(sample_p_hand_b)
        
        sample_p_hand_random = np.stack(sample_p_hand_random_list, axis=0)

        sample_p_hand = torch.cat([hand_verts_gt, torch.tensor(sample_p_hand_random).float().to(self.device)], dim=1).to(self.device)
        # write_ply("test_sample_p_hand.ply", points=sample_p_hand[0].detach().cpu().numpy())
        
        for b in range(B):
            shuffle_indice = torch.randperm(N_s).to(self.device)
            sample_p_hand[b] = sample_p_hand[b, shuffle_indice, :]


        hand_f = SDF(hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), self.mano_close_face.astype(np.float32))
        hand_gt_mesh = trimesh.Trimesh(vertices=hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), faces=self.mano_close_face.astype(np.int32))
        # hand_gt_mesh.export("test_gt_0.obj")
        
        hand_sdf_gt = torch.zeros((B, N_s), dtype=torch.float32).to(self.device)
        for sample_p_hand_b in sample_p_hand.detach().cpu().numpy().astype(np.float32):
            sdf_gt_b_np = -hand_f(sample_p_hand_b).astype(np.float32)
            hand_sdf_gt_b = torch.tensor(sdf_gt_b_np).to(self.device)
            hand_sdf_gt[b] = hand_sdf_gt_b
        
        return sample_p_hand, hand_sdf_gt

    
    def forward(self, pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand):
        ### Predict wnf for sample_p
        out_wnf, pc_feat_fused_grid, pred_flow = self.forward_obj(pc_1, pc_2, sample_p)
        
        ### Predict hand pose
        if self.hand_sdf_decoder is not None:
            # hand_sdf = self.forward_hand_sdf(pc_feat_fused_grid, sample_p_hand)
            hand_info = self.forward_hand_jtrs(pc_1, pc_2)
            hand_info = self.forward_hand_sdf(hand_info, sample_p_hand, pc_for_norm)
            return out_wnf, hand_info, pred_flow
        
        return out_wnf, None, pred_flow
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        B, N, C = pc_2.size()
        
        ### For testing the gradient
        # mano = torch.zeros_like(mano).to(self.device)
        
        sample_p_hand, hand_sdf_gt = self.generate_sample_p_hand(mano, pc_for_norm, sample_p.size(1))
        
        # ### For testing the gradient
        # hand_sdf_gt = torch.zeros_like(hand_sdf_gt).to(self.device)
        
        occ_pred, hand_info, pred_flow = self.forward(pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand)
        
        
        ### Total loss
        loss = 0
        
        ### SDF prediction loss
        if "sdf" in self.loss_items.keys():
            loss_occ = F.l1_loss(occ_pred, occ)
            loss += loss_occ
            self.log("occ_loss", loss_occ)

        ### Flow self-supervised loss
        if "flow" in self.loss_items.keys():
            # pc_2_corr = torch.einsum('bji,bik->bjk', pred_corr, pc_2.permute(0, 2, 1))
            # pc_1_corr = torch.einsum('bji,bik->bjk', pred_corr_back, pc_1.permute(0, 2, 1))
            # loss_flow = chamfer_distance((pc_1 + pred_flow).permute(0, 2, 1), pc_2_corr)
            loss_flow = chamfer_distance((pc_1.permute(0, 2, 1) + pred_flow).permute(0, 2, 1), pc_2)
            loss += loss_flow
            
            ### Backward flow prediction
            pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
            pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
            ### Correspondance Fusion feature from two pc
            pred_corr_back, pred_flow_back, pc_feat_fused_back = self.corr_fuser(pc_2, pc_1, pc_2_feat, pc_1_feat)
        
            # loss_flow_back = chamfer_distance((pc_2 + pred_flow_back).permute(0, 2, 1), pc_1_corr)
            loss_flow_back = chamfer_distance((pc_2.permute(0, 2, 1) + pred_flow_back).permute(0, 2, 1), pc_1)
            # loss += loss_flow_back
            
            self.log("flow_loss", loss_flow)
            
            
        
        if (hand_info is not None) and ("hand" in self.loss_items.keys()):
            
            mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
            
            hand_jtrs_pred_full = hand_info['jtrs']
            mano_hand_trans_pred = hand_info['hand_trans']
            mano_ik_pose_pred = hand_info['pose']
            hand_sdf = hand_info['sdf']
            
            ### Hand Joint loss
            mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
            hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
            
            hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
            
            loss_hand_joint = F.mse_loss(hand_jtrs_pred_full, hand_jtrs_gt)
            
            loss += loss_hand_joint
            self.log("hand_joint_loss", loss_hand_joint)
            
            write_ply("hand_joint.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
            write_ply("hand_joint_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())
            
            
            ### Ik pose
            # hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full[:, 1:, :], pc_for_norm)
            # hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred, self.device)
            
            # mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
            # mano_ik_pose_pred = mano_ik_pred_info['pose']


            # mano_ik_full_pose_pred = torch.cat([hand_jtrs_pred_full[:, 0, :], mano_ik_pose_pred], dim=1)
            
            mano_ik_full_pose_pred = torch.cat([mano_hand_trans_gt, mano_hand_rot_gt[:, :3], mano_ik_pose_pred[:, 3:]], dim=1)
            
            mano_info_pred = self.forward_manolayer(mano_ik_full_pose_pred, pc_for_norm)
            hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']
            
            
            ### Test, backward till 100 epochs
            # if self.current_epoch >= 100:
            #     # loss_hand_pose = F.mse_loss(mano_ik_pose_pred, mano_hand_rot_gt)
            #     # loss += loss_hand_pose
            #     # self.log("hand_pose_loss", loss_hand_pose)
                
            #     loss_sdf = F.l1_loss(hand_sdf, hand_sdf_gt)
            #     loss += loss_sdf
            #     self.log("hand_sdf_loss", loss_sdf)
                
            
            
            ### Test vis
            mano_ik_mesh = trimesh.Trimesh(vertices=hand_verts_pred[0].detach().cpu().numpy(), faces=self.mano_close_face)
            mano_ik_mesh.export("test_ik.obj")
            
            mano_gt_mesh = trimesh.Trimesh(vertices=hand_verts_gt[0].detach().cpu().numpy(), faces=self.mano_close_face)
            mano_gt_mesh.export("test_gt.obj")
            
            # mano_info_pred = ik_solver_mano(self.manolayer, hand_jtrs_gt, None)
            # pose_gt_ik = mano_info_pred['pose']
            # trans_gt_ik = mano_info_pred['global_trans'][:, 0, 3, :3]
            
            
            # loss_hand = self.loss_items['hand'] * F.mse_loss(mano_pred, mano)
            
            # mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
            # hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
            
            # mano_info_pred = self.forward_manolayer(mano_pred, pc_for_norm)
            # hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']
            
            # loss_hand_verts = self.loss_items['hand'] * F.mse_loss(hand_verts_pred, hand_verts_gt)
            # loss_hand_verts += self.loss_items['hand'] * F.mse_loss(hand_jtrs_pred, hand_jtrs_gt)
            
            # loss_hand += loss_hand_verts
            
            # self.log("hand_loss", loss_hand)
            
            # loss += loss_hand
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        ### For testing the gradient
        # mano = torch.zeros_like(mano).to(self.device)
        
        sample_p_hand, hand_sdf_gt = self.generate_sample_p_hand(mano, pc_for_norm, sample_p.size(1))
        
        occ_pred, hand_info, *_ = self.forward(pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand)
            
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        # write_ply("sample_p.ply", sample_p[0].detach().cpu().numpy())
        iou = compute_iou(occ1=occ.detach().cpu(), occ2=occ_pred.detach().cpu(), threshold=0.5)[0]
        print("Validation metric (iou):", iou)
        self.log("iou", iou)
        
        
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
        
        hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
        
        hand_jtr_cd = chamfer_distance(hand_jtrs_pred_full, hand_jtrs_gt)
        print("Validation metric (hand_jtr_cd):", hand_jtr_cd.item())
        self.log("hand_jtr_cd", hand_jtr_cd.item())
        
        # hand_sdf = torch.zeros_like(hand_sdf_gt).to(self.device)
        # iou_hand = compute_iou(occ1=hand_sdf_gt.detach().cpu(), occ2=hand_sdf.detach().cpu(), threshold=0.5)[0]
        # self.log("iou_hand", iou_hand)
        
        # write_ply("hand_joint_val.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
        # write_ply("hand_joint_val_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())

        return iou
    
    def vis_obj(self, pc_1, pc_2, sample_p_grid_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            occ_pred_list = []
            for sample_p in sample_p_grid_split:
                sample_p = sample_p.to(self.device)
                occ_pred, pc_feat_fused_grid, pred_flow = self.forward_obj(pc_1, pc_2, sample_p.unsqueeze(0))
                occ_pred_list.append(occ_pred)
        
        occ_pred = torch.cat(occ_pred_list, dim=1).to(self.device)
        
        occ_grid = occ_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, occ_grid, self.padding)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return occ_grid, vertices, faces, pc_feat_fused_grid
    
    def vis_hand_sdf_full(self, pc_1, pc_2, pc_for_norm, sample_p_grid_hand_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            hand_sdf_pred_list = []
            for sample_p in sample_p_grid_hand_split:
                sample_p = sample_p.to(self.device)
                hand_info = self.forward_hand_jtrs(pc_1, pc_2)
                hand_info = self.forward_hand_sdf(hand_info, sample_p.unsqueeze(0), pc_for_norm)
                
                hand_sdf = hand_info['sdf']
                hand_sdf_pred_list.append(hand_sdf)
        
        hand_sdf_pred = torch.cat(hand_sdf_pred_list, dim=1).to(self.device)
        
        hand_sdf_grid = hand_sdf_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, hand_sdf_grid, self.hand_padding)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return hand_sdf_grid, vertices, faces
    
    def vis_corr_flow(self, batch, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2 = pc_1.float(), pc_2.float()
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat, pc_1_feat_global, *_ = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat, pc_2_feat_global, *_ = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        return pc_1.permute(0, 2, 1) + pred_flow
    
    def vis_step(self, batch, batch_idx):
        ### Used on callback for visulization, or geo fit
        start_vis_time = time.time()
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, occ, mano, pc_for_norm = pc_1.float(), pc_2.float(), occ.float(), mano.float(), pc_for_norm.float()
        
        B, N, C = pc_2.size()
        sample_p_grid = self.box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_split = torch.split(sample_p_grid, self.vis_point_every_split)
        
        sample_p_grid_hand = self.hand_box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_hand_split = torch.split(sample_p_grid_hand, self.vis_point_every_split)
        
        ### For testing the gt
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        hand_f = SDF(hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), self.mano_close_face.astype(np.float32))
        sdf_grid = hand_f(sample_p_grid_hand.cpu().numpy().astype(np.float32)).astype(np.float32)
        sdf_grid = torch.tensor(-sdf_grid.reshape(self.nx, self.nx, self.nx), dtype=torch.float32).to(self.device)
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        hand_verts_gt, hand_faces_gt = self.extract_mesh(extractor, sdf_grid, self.hand_padding)
        hand_mesh_gt = trimesh.Trimesh(vertices=hand_verts_gt.cpu().numpy(), faces=hand_faces_gt.cpu().numpy())
        hand_mesh_gt.export("test_val_gt.obj")
        
        
        with torch.no_grad():
            occ_grid_list, verts_obj_list, face_obj_list = [], [], []
            verts_hand_list, face_hand_list = [], []
            for b in range(B):
                pc_1_batch, pc_2_batch = pc_1[b:b+1], pc_2[b:b+1]
                occ_grid, verts_obj, face_obj, pc_feat_fused_grid = self.vis_obj(pc_1_batch, pc_2_batch, sample_p_grid_split)
                
                hand_sdf_grid, verts_hand, face_hand = self.vis_hand_sdf_full(pc_1_batch, pc_2_batch, pc_for_norm, sample_p_grid_hand_split)
                
                verts_obj, face_obj = np.array(verts_obj, dtype=np.float32), np.array(face_obj, dtype=np.int32)
                verts_hand, face_hand = np.array(verts_hand, dtype=np.float32), np.array(face_hand, dtype=np.int32)

                occ_grid_list.append(occ_grid)
                verts_obj_list.append(verts_obj)
                face_obj_list.append(face_obj)
                
                verts_hand_list.append(verts_hand)
                face_hand_list.append(face_hand)
                
            if self.print_vis_time:
                print("Vis time: ", time.time() - start_vis_time)

            # if self.hand_tracker is not None:
            #     mano_pred, verts_hand, face_hand = self.vis_hand(pc_1_batch, pc_2_batch, pc_for_norm)
                
            #     ### Test if hand vis is correct
            #     # mano_pred = None
            #     # mano_info = self.forward_manolayer(mano_pred, pc_for_norm)
            #     # verts_hand, face_hand, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']
                
            #     verts_hand, face_hand = verts_hand.detach().cpu().numpy(), face_hand.cpu().numpy()
                
            # else:
            #     verts_hand, face_hand = None, None
            mano_pred = None
        return name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand_list, face_hand_list[0]
    
    
class SDFPipeline(BaseTrackingPipeline):
    def __init__(self,
                 training_param,
                 pointnet_param,
                 Corr_param,
                 grid_param,
                 unet3d_param,
                 decoder_param,
                 vis_param,
                 loss_items, 
                 hand_pose_cfg_file=None,
                 hand_pose_module_ckpt=None,
                 hand_feat_encoder_param=None,
                 hand_sdf_decoder_param=None,
                 vis_seq_list=None
                 ):
        super().__init__()

        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"

        ### Pointnet
        # self.pc_feature_encoder = PointNetfeat(**pointnet_param)
        self.pc_feature_encoder = PointNetPlus(**pointnet_param)
        
        ### Correspondance Fusion
        self.corr_fuser = CorrFlowFusion(**Corr_param)
        
        ### Grid feature Generator
        self.grid_dim = grid_param['grid_dim']
        self.reso_grid = grid_param['reso_grid']
        self.padding = grid_param['padding']
        
        ### Grid feature Sampler
        self.sample_mode = grid_param['sample_mode']
        
        self.unet3d = UNet3D(**unet3d_param)
        

        ### WNF Decoder, inherit from ConvOccNet
        self.decoder = WNFdecoder(**grid_param, **decoder_param)
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.nx = vis_param['nx']
        self.box_size = 1 + self.padding
        self.vis_point_every_split = vis_param['vis_point_every_split']
        self.print_vis_time = vis_param['print_time']
        self.vis_flow = vis_param['vis_flow']
        
        ### Loss items
        self.loss_items = loss_items
        
        ### Hand pose estimator
        if hand_pose_cfg_file is not None:
            with open(hand_pose_cfg_file, "r") as f:
                hand_pose_cfg = yaml.load(f, Loader=yaml.FullLoader)
                
            self.hand_pose_tracking_module = HandPoseTrackingModule(**hand_pose_cfg['trackingmodule'])
            print("Loading Hand pose estimation Module...")
            # self.hand_pose_tracking_module.load_model_only(hand_pose_module_ckpt)
            
            ### Closed Mano hand faces
            self.mano_close_face = self.hand_pose_tracking_module.mano_close_face
            self.manolayer = self.hand_pose_tracking_module.manolayer
            
            ### Freeze hand pose module
            for param in self.hand_pose_tracking_module.parameters():
                param.requires_grad = False

            
            ### Hand visual encoder
            self.pc_hand_feat_encoder = PointNetPlus(**hand_feat_encoder_param['pointnet_param'])
            self.hand_f_fuser = TransformerFusion(**hand_feat_encoder_param['transformer_param'])
            
            ### Hand SDF Decoder
            self.feat_mlp = MLPDecoder(**hand_sdf_decoder_param['feat_mlp_param'])
            self.hand_sdf_decoder = WNFdecoder(**grid_param, **hand_sdf_decoder_param['decoder_param'])

            ### Hand SDF Grid param
            self.hand_padding = hand_sdf_decoder_param['grid_param']['padding']
            self.hand_box_size = 1 + self.hand_padding
        else:
            self.hand_pose_tracking_module = None
            
            
    def forward_manolayer(self, hand_pose, pc_for_norm):
        B, _ = hand_pose.size()
        mano_hand_trans, mano_hand_rot = hand_pose[:, :3], hand_pose[:, 3:]
        mano_info = self.manolayer(mano_hand_rot)
        
        mano_verts, mano_jtrs, mano_transf, _ = mano_info
        
        mano_verts = rotate_hand_points_tensor(mano_verts, mano_hand_trans, self.device)
        mano_verts = norm_pointcloud_batch(mano_verts, pc_for_norm)
        mano_faces = self.manolayer.th_faces
        
        mano_jtrs = rotate_hand_points_tensor(mano_jtrs, mano_hand_trans, self.device)
        mano_jtrs = norm_pointcloud_batch(mano_jtrs, pc_for_norm)
        
        mano_trans_cat = torch.cat(
            [
                torch.cat([torch.zeros(B, 3, 3).to(self.device), mano_hand_trans.view(B, 3, -1)], dim=2),
                torch.zeros(B, 1, 4).to(self.device),
            ],
            dim=1,
        ).unsqueeze(1)

        mano_transf = mano_transf + mano_trans_cat
        
        mano_info = {'mano_verts': mano_verts, 
                     'mano_faces': mano_faces, 
                     'mano_jtrs': mano_jtrs,
                     'mano_transf': mano_transf}
        
        return mano_info
    
    
    def forward_obj(self, pc_1, pc_2, sample_p):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_fused_grid = self.generate_grid_features(pc_2, pc_feat_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_feat = self.sample_grid_feature(sample_p, pc_feat_fused_grid)
        
        ### Decode wnf from sample_p_feat
        out_wnf = self.decoder(sample_p, sample_p_feat)
        
        return out_wnf, pc_feat_fused_grid, pred_flow
    
    
    def forward_hand(self, pc_1, pc_2, pc_for_norm, sample_p_hand):
        B, N_s, _ = sample_p_hand.size()
        
        hand_info = self.hand_pose_tracking_module.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        mano_ik_pose_pred = hand_info['pose']
        
        ### Hand visual feature encode
        ### Pointnet encode feature from two pc
        pc_1_feat_hand = self.pc_hand_feat_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat_hand = self.pc_hand_feat_encoder(pc_2.permute(0, 2, 1))
        
        ### Transformer Fusion feature from two pc
        pc_feat_hand_fused = self.hand_f_fuser(pc_2_feat_hand, pc_2, pc_1_feat_hand, pc_1)
        
        ### Generate grid features from pc_2 and fused_feat
        pc_feat_hand_fused_grid = self.generate_grid_features(pc_2, pc_feat_hand_fused)
        
        ### Sample feature for sample points from grid_feat
        sample_p_hand_feat = self.sample_grid_feature(sample_p_hand, pc_feat_hand_fused_grid)
        
        ### Hand pose feat encode
        feat_pose = self.feat_mlp(mano_ik_pose_pred)
        feat_p_sdf = feat_pose.unsqueeze(1).repeat(1, N_s, 1)
        
        ### SDF Feat
        feat_hand_sdf = sample_p_hand_feat + feat_p_sdf
        
        ### SDF prediction
        hand_sdf = self.hand_sdf_decoder(sample_p_hand, feat_hand_sdf)
        
        hand_info['sdf'] = hand_sdf
        return hand_info

    
    def generate_sample_p_hand(self, mano, pc_for_norm, N_s):
        B = mano.size(0)
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        sample_p_hand_random_list = []
        for b in range(B):
            sample_p_hand_b = np.array(np.array([-self.hand_box_size/2, -self.hand_box_size/2, -self.hand_box_size/2]) + np.random.rand(N_s - hand_verts_gt.size(1), 3)*self.hand_box_size, dtype=np.float32)
            sample_p_hand_random_list.append(sample_p_hand_b)
        
        sample_p_hand_random = np.stack(sample_p_hand_random_list, axis=0)

        sample_p_hand = torch.cat([hand_verts_gt, torch.tensor(sample_p_hand_random).float().to(self.device)], dim=1).to(self.device)
        # write_ply("test_sample_p_hand.ply", points=sample_p_hand[0].detach().cpu().numpy())
        
        for b in range(B):
            shuffle_indice = torch.randperm(N_s).to(self.device)
            sample_p_hand[b] = sample_p_hand[b, shuffle_indice, :]


        hand_f = SDF(hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), self.mano_close_face.astype(np.float32))
        hand_gt_mesh = trimesh.Trimesh(vertices=hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), faces=self.mano_close_face.astype(np.int32))
        # hand_gt_mesh.export("test_gt_0.obj")
        
        hand_sdf_gt = torch.zeros((B, N_s), dtype=torch.float32).to(self.device)
        for sample_p_hand_b in sample_p_hand.detach().cpu().numpy().astype(np.float32):
            sdf_gt_b_np = -hand_f(sample_p_hand_b).astype(np.float32)
            hand_sdf_gt_b = torch.tensor(sdf_gt_b_np).to(self.device)
            hand_sdf_gt[b] = hand_sdf_gt_b
        
        return sample_p_hand, hand_sdf_gt

    
    def forward(self, pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand):
        ### Predict wnf for sample_p
        out_wnf, pc_feat_fused_grid, pred_flow = self.forward_obj(pc_1, pc_2, sample_p)
        
        ### Predict hand pose
        if self.hand_sdf_decoder is not None:
            # hand_info = self.forward_hand(pc_1, pc_2, pc_for_norm, sample_p_hand)
            hand_info = None
            return out_wnf, hand_info, pred_flow
        
        return out_wnf, None, pred_flow
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        B, N, C = pc_2.size()
        
        ### For testing the gradient
        # mano = torch.zeros_like(mano).to(self.device)
        
        sample_p_hand, hand_sdf_gt = self.generate_sample_p_hand(mano, pc_for_norm, sample_p.size(1))
        
        # ### For testing the gradient
        # hand_sdf_gt = torch.zeros_like(hand_sdf_gt).to(self.device)
        
        occ_pred, hand_info, pred_flow = self.forward(pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand)
        
        
        ### Total loss
        loss = 0
        
        ### SDF prediction loss
        if "sdf" in self.loss_items.keys():
            loss_occ = F.l1_loss(occ_pred, occ)
            loss += loss_occ
            self.log("occ_loss", loss_occ)

        ### Flow self-supervised loss
        if "flow" in self.loss_items.keys():
            # pc_2_corr = torch.einsum('bji,bik->bjk', pred_corr, pc_2.permute(0, 2, 1))
            # pc_1_corr = torch.einsum('bji,bik->bjk', pred_corr_back, pc_1.permute(0, 2, 1))
            # loss_flow = chamfer_distance((pc_1 + pred_flow).permute(0, 2, 1), pc_2_corr)
            loss_flow = chamfer_distance((pc_1.permute(0, 2, 1) + pred_flow).permute(0, 2, 1), pc_2)
            loss += loss_flow
            
            ### Backward flow prediction
            pc_1_feat = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
            pc_2_feat = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
            ### Correspondance Fusion feature from two pc
            pred_corr_back, pred_flow_back, pc_feat_fused_back = self.corr_fuser(pc_2, pc_1, pc_2_feat, pc_1_feat)
        
            # loss_flow_back = chamfer_distance((pc_2 + pred_flow_back).permute(0, 2, 1), pc_1_corr)
            loss_flow_back = chamfer_distance((pc_2.permute(0, 2, 1) + pred_flow_back).permute(0, 2, 1), pc_1)
            loss += loss_flow_back
            
            self.log("flow_loss", loss_flow)
            
            
        
        if (hand_info is not None) and ("hand" in self.loss_items.keys()):
            
            mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
            
            hand_jtrs_pred_full = hand_info['jtrs']
            mano_hand_trans_pred = hand_info['hand_trans']
            mano_ik_pose_pred = hand_info['pose']
            hand_sdf = hand_info['sdf']
            
            ### Hand Joint loss
            mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
            hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
            
            hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
            
            # loss_hand_joint = F.mse_loss(hand_jtrs_pred_full, hand_jtrs_gt)
            # loss += loss_hand_joint
            # self.log("hand_joint_loss", loss_hand_joint)
            
            # write_ply("hand_joint.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
            # write_ply("hand_joint_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())


            # mano_ik_full_pose_pred = torch.cat([hand_jtrs_pred_full[:, 0, :], mano_ik_pose_pred], dim=1)
            mano_ik_full_pose_pred = torch.cat([mano_hand_trans_gt, mano_hand_rot_gt[:, :3], mano_ik_pose_pred[:, 3:]], dim=1)
            
            mano_info_pred = self.forward_manolayer(mano_ik_full_pose_pred, pc_for_norm)
            hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']
            
            
            ### Test, backward till 100 epochs
            loss_sdf = F.l1_loss(hand_sdf, hand_sdf_gt)
            loss += loss_sdf
            self.log("hand_sdf_loss", loss_sdf)
                
            
            
            ### Test vis
            # mano_ik_mesh = trimesh.Trimesh(vertices=hand_verts_pred[0].detach().cpu().numpy(), faces=self.mano_close_face)
            # mano_ik_mesh.export("test_ik.obj")
            
            # mano_gt_mesh = trimesh.Trimesh(vertices=hand_verts_gt[0].detach().cpu().numpy(), faces=self.mano_close_face)
            # mano_gt_mesh.export("test_gt.obj")
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()

        ### For testing the gradient
        # mano = torch.zeros_like(mano).to(self.device)
        
        sample_p_hand, hand_sdf_gt = self.generate_sample_p_hand(mano, pc_for_norm, sample_p.size(1))
        
        occ_pred, hand_info, *_ = self.forward(pc_1, pc_2, pc_for_norm, sample_p, sample_p_hand)
            
        
        # write_ply("sample_p.ply", sample_p[0].detach().cpu().numpy())
        iou = compute_iou(occ1=occ.detach().cpu(), occ2=occ_pred.detach().cpu(), threshold=0.5)[0]
        print("Validation metric (iou):", iou)
        self.log("iou", iou)
        
        ### Hand metrics
        # hand_jtrs_pred_full = hand_info['jtrs']
        # mano_hand_trans_pred = hand_info['hand_trans']
        
        # mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        # hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        # mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
        
        # hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
        
        # hand_jtr_cd = chamfer_distance(hand_jtrs_pred_full, hand_jtrs_gt)
        # self.log("hand_jtr_cd", hand_jtr_cd.item())
        
        # hand_sdf = torch.zeros_like(hand_sdf_gt).to(self.device)
        # iou_hand = compute_iou(occ1=hand_sdf_gt.detach().cpu(), occ2=hand_sdf.detach().cpu(), threshold=0.5)[0]
        # self.log("iou_hand", iou_hand)
        
        # write_ply("hand_joint_val.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
        # write_ply("hand_joint_val_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())

        return iou
    
    def vis_obj(self, pc_1, pc_2, sample_p_grid_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            occ_pred_list = []
            for sample_p in sample_p_grid_split:
                sample_p = sample_p.to(self.device)
                occ_pred, pc_feat_fused_grid, pred_flow = self.forward_obj(pc_1, pc_2, sample_p.unsqueeze(0))
                occ_pred_list.append(occ_pred)
        
        occ_pred = torch.cat(occ_pred_list, dim=1).to(self.device)
        
        occ_grid = occ_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, occ_grid, self.padding)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return occ_grid, vertices, faces, pc_feat_fused_grid
    
    def vis_hand_sdf(self, pc_1, pc_2, pc_for_norm, sample_p_grid_hand_split):
        ### Used on callback for visulization, or geo fit, with batchsize 1
        with torch.no_grad():
            hand_sdf_pred_list = []
            for sample_p_hand in sample_p_grid_hand_split:
                sample_p_hand = sample_p_hand.to(self.device)
                hand_info = self.forward_hand(pc_1, pc_2, pc_for_norm, sample_p_hand.unsqueeze(0))
                
                hand_sdf = hand_info['sdf']
                hand_sdf_pred_list.append(hand_sdf)
        
        hand_sdf_pred = torch.cat(hand_sdf_pred_list, dim=1).to(self.device)
        
        hand_sdf_grid = hand_sdf_pred.reshape(self.nx, self.nx, self.nx)
        
        ### Skimage marching cubes, deprecated
        # vertices, faces, normals, _ = measure.marching_cubes(occ_grid.cpu().numpy(), gradient_direction='ascent')
        # vertices -= np.array([self.nx/2, self.nx/2, self.nx/2], dtype=np.float32)
        # vertices *= (1+self.padding)/self.nx
        
        ### Diso for mesh recon, used on sdf data
        extractor = DiffDMC(dtype=torch.float32).to(self.device)
        verts_diso, faces_diso = self.extract_mesh(extractor, hand_sdf_grid, self.hand_padding)
        
        vertices, faces = verts_diso.cpu().numpy(), faces_diso.cpu().numpy()

        return hand_sdf_grid, vertices, faces
    
    def vis_corr_flow(self, batch, batch_idx):
        # in lightning, forward defines the prediction/inference actions
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2 = pc_1.float(), pc_2.float()
        B, N, C = pc_2.size()
        
        ### Pointnet encode feature from two pc
        pc_1_feat = self.pc_feature_encoder(pc_1.permute(0, 2, 1))
        pc_2_feat = self.pc_feature_encoder(pc_2.permute(0, 2, 1))
        
        ### Correspondance Fusion feature from two pc
        pred_corr, pred_flow, pc_feat_fused = self.corr_fuser(pc_1, pc_2, pc_1_feat, pc_2_feat)
        
        return pc_1.permute(0, 2, 1) + pred_flow
    
    def vis_step(self, batch, batch_idx):
        ### Used on callback for visulization, or geo fit
        start_vis_time = time.time()
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, occ, mano, pc_for_norm = pc_1.float(), pc_2.float(), occ.float(), mano.float(), pc_for_norm.float()
        
        B, N, C = pc_2.size()
        sample_p_grid = self.box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_split = torch.split(sample_p_grid, self.vis_point_every_split)
        
        sample_p_grid_hand = self.hand_box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (self.nx,)*3
            )
        sample_p_grid_hand_split = torch.split(sample_p_grid_hand, self.vis_point_every_split)
        
        ### For testing the gt
        # mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        # hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        # hand_f = SDF(hand_verts_gt[0].detach().cpu().numpy().astype(np.float32), self.mano_close_face.astype(np.float32))
        # sdf_grid = hand_f(sample_p_grid_hand.cpu().numpy().astype(np.float32)).astype(np.float32)
        # sdf_grid = torch.tensor(-sdf_grid.reshape(self.nx, self.nx, self.nx), dtype=torch.float32).to(self.device)
        # extractor = DiffDMC(dtype=torch.float32).to(self.device)
        # hand_verts_gt, hand_faces_gt = self.extract_mesh(extractor, sdf_grid, self.hand_padding)
        # hand_mesh_gt = trimesh.Trimesh(vertices=hand_verts_gt.cpu().numpy(), faces=hand_faces_gt.cpu().numpy())
        # hand_mesh_gt.export("test_val_gt.obj")
        
        
        with torch.no_grad():
            occ_grid_list, verts_obj_list, face_obj_list = [], [], []
            verts_hand_list, face_hand_list = [], []
            for b in range(B):
                pc_1_batch, pc_2_batch = pc_1[b:b+1], pc_2[b:b+1]
                occ_grid, verts_obj, face_obj, pc_feat_fused_grid = self.vis_obj(pc_1_batch, pc_2_batch, sample_p_grid_split)
                
                # hand_sdf_grid, verts_hand, face_hand = self.vis_hand_sdf(pc_1_batch, pc_2_batch, pc_for_norm, sample_p_grid_hand_split)
                
                verts_obj, face_obj = np.array(verts_obj, dtype=np.float32), np.array(face_obj, dtype=np.int32)
                # verts_hand, face_hand = np.array(verts_hand, dtype=np.float32), np.array(face_hand, dtype=np.int32)
                verts_hand, face_hand = None, None

                occ_grid_list.append(occ_grid)
                verts_obj_list.append(verts_obj)
                face_obj_list.append(face_obj)
                
                verts_hand_list.append(verts_hand)
                face_hand_list.append(face_hand)
                
            if self.print_vis_time:
                print("Vis time: ", time.time() - start_vis_time)

            # if self.hand_tracker is not None:
            #     mano_pred, verts_hand, face_hand = self.vis_hand(pc_1_batch, pc_2_batch, pc_for_norm)
                
            #     ### Test if hand vis is correct
            #     # mano_pred = None
            #     # mano_info = self.forward_manolayer(mano_pred, pc_for_norm)
            #     # verts_hand, face_hand, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']
                
            #     verts_hand, face_hand = verts_hand.detach().cpu().numpy(), face_hand.cpu().numpy()
                
            # else:
            #     verts_hand, face_hand = None, None
            mano_pred = None
        return name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand_list, face_hand_list[0]