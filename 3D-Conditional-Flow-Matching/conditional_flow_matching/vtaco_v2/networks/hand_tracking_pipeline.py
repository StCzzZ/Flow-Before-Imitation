import os
import time
import numpy as np
import trimesh
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from networks.pointnet.pointnet_plus import PointNetPlus
from networks.layer import EquivariantLayer, MLPDecoder
from networks.transformer import TransformerFusion
from components.manopth.manopth.manolayer_old import ManoLayer
from components.manopth.manopth.inverse_kinematics import ik_solver_mano

from common.pointcloud_utils import norm_pointcloud, norm_pointcloud_batch, rotate_hand_points_tensor, chamfer_distance,\
    denorm_pointcloud_batch, inverse_rotate_hand_points_tensor
from common.io_utils import write_ply

### Load the checkpoint with this function
from pytorch_lightning.utilities.cloud_io import load as pl_load

torch.set_default_dtype(torch.float32)

class HandPoseTrackingModule(pl.LightningModule):
    def __init__(self,
                 assets_dir,
                 training_param,
                 vis_param,
                 loss_items, 
                 hand_decoder_param,
                 vis_seq_list=None
                 ):
        super().__init__()
        
        ### Closed Mano hand faces
        self.mano_close_face = np.load(os.path.join(assets_dir, "closed_fmano.npy")).astype(np.int32)
        
        ### Training parameters
        self.lr = training_param['lr']
        self.optim = training_param['optimizer']
        
        assert self.optim in ["SGD", "Adam"], "Optimizer not supported"
        
        ### Visualization param
        self.vis_every = vis_param['vis_every']
        self.print_vis_time = vis_param['print_time']
        self.vis_dir = os.path.join(os.getcwd(), "vis")
        
        ### Loss items
        self.loss_items = loss_items
        
        ### Hand pose estimator
        self.num_kp = hand_decoder_param['num_kp']
        self.dense_soft_factor = hand_decoder_param['dense_soft_factor']
        layer_in_dim = hand_decoder_param['transformer_param']['fea_channels'][-1]
        layer_hidden_dim = hand_decoder_param['layer_hidden_dim']
        self.use_attention = hand_decoder_param['use_attention']
        
        self.pc_hand_feat_encoder = PointNetPlus(**hand_decoder_param['pointnet_param'])
        
        self.hand_f_fuser = TransformerFusion(**hand_decoder_param['transformer_param'])
        
        self.cls_1 = EquivariantLayer(layer_in_dim, layer_hidden_dim, activation='relu', normalization='batch')

        self.kp_1 = EquivariantLayer(layer_in_dim, layer_hidden_dim, activation='relu', normalization='batch')
        self.kp_2 = EquivariantLayer(layer_hidden_dim, 3 * self.num_kp, activation=None, normalization=None)
        
        if self.use_attention:
            self.kp_att_1 = EquivariantLayer(layer_in_dim, layer_hidden_dim, activation='relu', normalization='batch')
            self.kp_att_2 = EquivariantLayer(layer_hidden_dim, self.num_kp, activation=None, normalization=None)
            
            self.kp_att_3 = EquivariantLayer(layer_in_dim, layer_hidden_dim, activation='relu', normalization='batch')
            self.kp_att_4 = EquivariantLayer(layer_hidden_dim, self.num_kp, activation=None, normalization=None)
                    
        self.manolayer = ManoLayer(**hand_decoder_param['manolayer_param'])
        
        
        ### Visualization list
        self.vis_seq_list = vis_seq_list
    
    def load_model_only(self, chkpt_path):
        ckpt = pl_load(chkpt_path)
        self.load_state_dict(ckpt['state_dict'])
        
        
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
    
    def forward_hand_jtrs_origin(self, pc_1, pc_2, pc_for_norm):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        feat = None
        pts = pc_2
        
        pc_1_feat = self.pc_hand_feat_encoder(pc_1.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        pc_2_feat = self.pc_hand_feat_encoder(pc_2.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        
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
        
        ### Ik pose
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        ### Denorm estimated root trans
        mano_hand_trans_pred_denormed = denorm_pointcloud_batch(mano_hand_trans_pred.unsqueeze(1), pc_for_norm).squeeze(1)
        
        hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full[:, 1:, :], pc_for_norm)
        hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred_denormed, self.device)
        
        mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
        mano_ik_pose_pred = mano_ik_pred_info['pose']
        
        hand_info['pose'] = mano_ik_pose_pred
        
        return hand_info

    def forward_hand_jtrs_jtrs2wrist(self, pc_1, pc_2, pc_for_norm):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        feat = None
        pts = pc_2
        
        pc_1_feat = self.pc_hand_feat_encoder(pc_1.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        pc_2_feat = self.pc_hand_feat_encoder(pc_2.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        
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
        mano_hand_trans = self.mlp_wrist(pred_kp.reshape(B, -1)).squeeze(1)

        pred_kp = torch.cat([mano_hand_trans.unsqueeze(1), pred_kp], dim=1)
        hand_info = {'jtrs': pred_kp, 
                     'hand_trans': mano_hand_trans}
        
        ### Ik pose
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        ### Denorm estimated root trans
        mano_hand_trans_pred_denormed = denorm_pointcloud_batch(mano_hand_trans_pred.unsqueeze(1), pc_for_norm).squeeze(1)
        
        hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full[:, 1:, :], pc_for_norm)
        hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred_denormed, self.device)
        
        mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
        mano_ik_pose_pred = mano_ik_pred_info['pose']
        
        hand_info['pose'] = mano_ik_pose_pred
        
        return hand_info
    
    def forward_hand_jtrs_maxpool(self, pc_1, pc_2, pc_for_norm):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        feat = None
        pts = pc_2
        
        pc_1_feat = self.pc_hand_feat_encoder(pc_1.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        pc_2_feat = self.pc_hand_feat_encoder(pc_2.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        
        ### Transformer Fusion feature from two pc
        dense_pts_feat = self.hand_f_fuser(pc_2_feat, pc_2, pc_1_feat, pc_1).transpose(1, 2).contiguous()  # (bs, 128, n)
        
        ### Predict wrist pose from dense pts feat
        dense_wrist_feat = self.conv_wrist_2(self.conv_wrist_1(dense_pts_feat)).contiguous()  # (bs, 3, n)

        mano_hand_trans = torch.max(dense_wrist_feat, dim=-1)[0]  # (bs, 3)

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

        pred_kp = torch.cat([mano_hand_trans.unsqueeze(1), pred_kp], dim=1)

        hand_info = {'jtrs': pred_kp, 
                     'hand_trans': mano_hand_trans}
        
        ### Ik pose
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        ### Denorm estimated root trans
        mano_hand_trans_pred_denormed = denorm_pointcloud_batch(mano_hand_trans_pred.unsqueeze(1), pc_for_norm).squeeze(1)
        
        hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full[:, 1:, :], pc_for_norm)
        hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred_denormed, self.device)
        
        mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
        mano_ik_pose_pred = mano_ik_pred_info['pose']
        
        hand_info['pose'] = mano_ik_pose_pred
        
        return hand_info
    
    def forward_hand_jtrs(self, pc_1, pc_2, pc_for_norm):
        # in lightning, forward defines the prediction/inference actions
        B, N, C = pc_2.size()
        
        feat = None
        pts = pc_2
        
        pc_1_feat = self.pc_hand_feat_encoder(pc_1.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        pc_2_feat = self.pc_hand_feat_encoder(pc_2.permute(0, 2, 1), feat).contiguous()  # (bs, 128, n)
        
        ### Transformer Fusion feature from two pc
        dense_pts_feat = self.hand_f_fuser(pc_2_feat, pc_2, pc_1_feat, pc_1).transpose(1, 2).contiguous()  # (bs, 128, n)

        ### Predict and vote hand keypoints offset
        dense_kp_offset = self.kp_2(self.kp_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, M*3)
        dense_kp_offset = dense_kp_offset.reshape(B, N, -1, 3)  # (bs, n, M, 3)
        
        if self.use_attention:
            dense_kp_off_score = self.kp_att_2(self.kp_att_1(dense_pts_feat))\
                .transpose(1, 2).contiguous().reshape(B, N, -1, 1)  # (bs, n, M, 1)
            dense_kp_off_prob = F.softmax(dense_kp_off_score * self.dense_soft_factor, dim=1)  # (bs, n, M, 1)
        else:
            dense_kp_off_prob = torch.ones_like(dense_kp_coords) / N
            
        pred_kp_offset = torch.sum(dense_kp_off_prob * dense_kp_offset, dim=1, keepdim=True)  # (bs, 1, M, 3)
        pred_kp_offset = pred_kp_offset.squeeze(1)  # (bs, M, 3)
        
        ### Predict and vote hand keypoints coords from pc_2
        dense_kp_coords = pts.reshape(B, N, 1, 3).expand(B, N, self.num_kp, 3)
        if self.use_attention:
            dense_kp_score = self.kp_att_4(self.kp_att_3(dense_pts_feat))\
                .transpose(1, 2).contiguous().reshape(B, N, -1, 1)  # (bs, n, M, 1)
            dense_kp_prob = F.softmax(dense_kp_score * self.dense_soft_factor, dim=1)  # (bs, n, M, 1)
        else:
            dense_kp_prob = torch.ones_like(dense_kp_coords) / N
        
        pred_kp_coords = torch.sum(dense_kp_prob * dense_kp_coords, dim=1, keepdim=True)  # (bs, 1, M, 3)
        pred_kp_coords = pred_kp_coords.squeeze(1)  # (bs, M, 3)
        
        pred_kp = pred_kp_coords + pred_kp_offset
        
        mano_hand_trans = pred_kp[:, 0, :].squeeze(1)

        hand_info = {'jtrs': pred_kp, 
                     'hand_trans': mano_hand_trans}
        
        ### Ik pose
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        ### Denorm estimated root trans
        mano_hand_trans_pred_denormed = denorm_pointcloud_batch(mano_hand_trans_pred.unsqueeze(1), pc_for_norm).squeeze(1)
        
        hand_jtrs_pred_full_denormed = denorm_pointcloud_batch(hand_jtrs_pred_full, pc_for_norm)
        hand_jtrs_pred_full_recover = inverse_rotate_hand_points_tensor(hand_jtrs_pred_full_denormed, mano_hand_trans_pred_denormed, self.device)
        
        mano_ik_pred_info = ik_solver_mano(self.manolayer, hand_jtrs_pred_full_recover, None)
        mano_ik_pose_pred = mano_ik_pred_info['pose']
        
        hand_info['pose'] = mano_ik_pose_pred
        
        return hand_info
        
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        B, N, C = pc_2.size()
        
        ### For testing the gradient
        # mano = torch.zeros_like(mano).to(self.device)
        
        # ### For testing the gradient
        # hand_sdf_gt = torch.zeros_like(hand_sdf_gt).to(self.device)
        
        # hand_info = self.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        
        ### Test different method
        hand_info = self.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        
        ### Total loss
        loss = 0
            
        mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
        
        ### Norm hand_trans
        mano_hand_trans_gt_normed = norm_pointcloud_batch(mano_hand_trans_gt.unsqueeze(1), pc_for_norm).squeeze(1)
        
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        mano_ik_pose_pred = hand_info['pose']
        
        ### Hand Joint loss
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        # hand_jtrs_gt = torch.cat([mano_hand_trans_gt_normed.unsqueeze(1), hand_jtrs_gt], dim=1)
        
        loss_hand_joint = F.mse_loss(hand_jtrs_pred_full, hand_jtrs_gt)
        
        loss += loss_hand_joint
        self.log("hand_joint_loss", loss_hand_joint)
        
        write_ply("hand_joint.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
        write_ply("hand_joint_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())


        # mano_ik_full_pose_pred = torch.cat([hand_jtrs_pred_full[:, 0, :], mano_ik_pose_pred], dim=1)
        
        mano_ik_full_pose_pred = torch.cat([mano_hand_trans_gt, mano_ik_pose_pred], dim=1)
        
        mano_info_pred = self.forward_manolayer(mano_ik_full_pose_pred, pc_for_norm)
        hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']

        if (self.current_epoch % self.vis_every == 0) and (self.current_epoch != 0):
            for b in range(B):
                seq_name_b = name[b][:-9]
                if seq_name_b in self.vis_seq_list:
                    ### Test vis
                    mesh_obj_name = "{:03d}_{}_hand.obj".format(self.current_epoch, name[b])
                    
                    mano_ik_mesh = trimesh.Trimesh(vertices=hand_verts_pred[0].detach().cpu().numpy(), faces=self.mano_close_face)
                    mano_ik_mesh.export(os.path.join(self.vis_dir, mesh_obj_name))
                    
                    mano_gt_mesh = trimesh.Trimesh(vertices=hand_verts_gt[0].detach().cpu().numpy(), faces=self.mano_close_face)
                    mano_gt_mesh.export(os.path.join(self.vis_dir, mesh_obj_name.replace(".obj", "_gt.obj")))
        

        return loss
    
    def validation_step(self, batch, batch_idx):
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, sample_p, occ, mano = pc_1.float(), pc_2.float(), sample_p.float(), occ.float(), mano.float()
        
        ### Test different method
        hand_info = self.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
            
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        
        mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
        hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
        
        mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]
        
        ### Norm hand_trans
        # mano_hand_trans_gt_normed = norm_pointcloud_batch(mano_hand_trans_gt.unsqueeze(1), pc_for_norm).squeeze(1)
        
        # hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
        
        hand_jtr_cd = chamfer_distance(hand_jtrs_pred_full, hand_jtrs_gt)
        print("Validation metric (hand_jtr_cd):", hand_jtr_cd.item())
        self.log("hand_jtr_cd", hand_jtr_cd.item())
        
        
        # write_ply("hand_joint_val.ply", points=hand_jtrs_pred_full[0].detach().cpu().numpy())
        # write_ply("hand_joint_val_gt.ply", points=hand_jtrs_gt[0].detach().cpu().numpy())

        return hand_jtr_cd
    
    def vis_hand_jtrs(self, pc_1, pc_2, pc_for_norm):
        # hand_info = self.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        
        ### Test different method
        hand_info = self.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        
        hand_jtrs_pred_full = hand_info['jtrs']
        mano_hand_trans_pred = hand_info['hand_trans']
        mano_ik_pose_pred = hand_info['pose']
        
        mano_hand_trans_pred_denormed = denorm_pointcloud_batch(mano_hand_trans_pred.unsqueeze(1), pc_for_norm).squeeze(1)

        mano_ik_full_pose_pred = torch.cat([mano_hand_trans_pred_denormed, mano_ik_pose_pred], dim=1)
        
        # mano_ik_full_pose_pred = torch.cat([mano_hand_trans_gt, mano_hand_rot_gt[:, :3], mano_ik_pose_pred[:, 3:]], dim=1)
        
        mano_info_pred = self.forward_manolayer(mano_ik_full_pose_pred, pc_for_norm)
        hand_verts_pred, hand_faces_pred, hand_jtrs_pred = mano_info_pred['mano_verts'], mano_info_pred['mano_faces'], mano_info_pred['mano_jtrs']
        
        return hand_jtrs_pred_full, hand_verts_pred, hand_faces_pred

    
    def vis_step(self, batch, batch_idx):
        ### Used on callback for visulization, or geo fit
        start_vis_time = time.time()
        
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, *_ = batch.values()
        pc_1, pc_2, occ, mano, pc_for_norm = pc_1.float(), pc_2.float(), occ.float(), mano.float(), pc_for_norm.float()
        
        B, N, C = pc_2.size()
        
        with torch.no_grad():
            mano_hand_trans_gt, mano_hand_rot_gt = mano[:, :3], mano[:, 3:]

            mano_info_gt = self.forward_manolayer(mano, pc_for_norm)
            hand_verts_gt, hand_faces_gt, hand_jtrs_gt = mano_info_gt['mano_verts'], mano_info_gt['mano_faces'], mano_info_gt['mano_jtrs']
            
            hand_jtrs_gt = torch.cat([mano_hand_trans_gt.unsqueeze(1), hand_jtrs_gt], dim=1)
            
            hand_jtrs_pred_full, hand_verts_pred, hand_faces_pred = self.vis_hand_jtrs(pc_1, pc_2, pc_for_norm)
            
            if self.print_vis_time:
                print("Vis time: ", time.time() - start_vis_time)

        return name, hand_jtrs_pred_full, hand_verts_pred, hand_faces_pred, hand_jtrs_gt, hand_verts_gt, hand_faces_gt
    
    def configure_optimizers(self):
        if self.optim == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer