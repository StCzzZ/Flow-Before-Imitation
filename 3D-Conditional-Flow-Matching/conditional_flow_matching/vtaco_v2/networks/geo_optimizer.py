import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d
import trimesh
from diso import DiffDMC

from common.pointcloud_utils import chamfer_distance, norm_pointcloud_batch, find_points_vec_dist_mask, rotate_hand_points_tensor
from common.mesh_utils import verts_faces_to_o3dmesh, compute_mesh_normals, compute_mesh_normals_batch
from common.io_utils import read_sensor_point_idx, write_ply
from common.quat_utils import *

from components.manopth.manopth.axislayer import AxisLayer
from components.handloss import HandLoss

class GeOptimizer():
    def __init__(self, pipeline_model, 
                       hand_model, 
                       assets_dir,
                       sensor_sample_mode,
                       threshold_dist,
                       k_repl,
                       optimize_item,
                       loss_item,
                       max_steps,
                       vis_every,
                       lr=0.01):
        ### Load pipeline models
        self.pipeline_model = pipeline_model
        self.hand_model = hand_model
        
        ### GeO Parameters
        self.assets_dir = assets_dir
        self.sensor_sample_mode = sensor_sample_mode
        self.threshold_dist = threshold_dist
        self.k_repl = k_repl
        self.optimize_item = optimize_item
        self.loss_item = loss_item
        
        self.lr = lr
        self.max_steps = max_steps
        self.vis_every = vis_every
        self.device = self.pipeline_model.device
        
        ### Load sensor-sample-required tensors
        sensor_idx_json = os.path.join(assets_dir, f"sensor_idx.json")
        _, self.region_name, self.region_idx = read_sensor_point_idx(sensor_idx_json, mode=sensor_sample_mode)
        
        hand_flat_mesh = trimesh.load_mesh(os.path.join(assets_dir, 'mano_hand_flat.obj'))
        self.hand_flat_faces = torch.tensor(hand_flat_mesh.faces, dtype=torch.int64).to(self.device)
        # faces indices
        self.hand_face_indices = torch.load(os.path.join(assets_dir, f"face_indices_{sensor_sample_mode}.pt")).long().to(self.device)
        # faces norm
        self.hand_norm_off = torch.load(os.path.join(assets_dir,f"norm_off_{sensor_sample_mode}.pt")).float().requires_grad_().to(self.device)
        # barycenter
        self.hand_bary_coords = torch.load(os.path.join(assets_dir, f"bary_coords_{sensor_sample_mode}.pt")).float().requires_grad_().to(self.device)
        
        
        ### Hand regularization loss
        self.axis_layer = AxisLayer().to(self.device)
        
        
    
    def fit_batch(self, batch, batch_idx):
        ### Param to update: vertices of obj, mano poses
        # Have to put them on device...
        for key, value in batch.items():
            try:
                batch[key] = value.float().to(self.device)
            except:
                pass
            
        name, pc_1, pc_2, sample_p, occ, mano, pc_for_norm, sensor_forces, p_obj_gt, *_ = batch.values()
        B, N, C = pc_1.size()
        
        name, occ_grid_list, verts_obj_list, face_obj_list, mano_pred, verts_hand, face_hand = self.pipeline_model.vis_step(batch, batch_idx)
        
        hand_info = self.hand_model.forward_hand_jtrs(pc_1, pc_2, pc_for_norm)
        mano_pred = hand_info['pose']
        mano_hand_trans_gt = mano[:, :3]
        mano_pred = torch.cat([mano_hand_trans_gt, mano_pred], dim=1)

        verts_obj = torch.tensor(np.array([verts_obj_list[i].copy() for i in range(len(verts_obj_list))]), dtype=torch.float32)
        faces_obj = torch.tensor(np.array([face_obj_list[i].copy() for i in range(len(verts_obj_list))]), dtype=torch.int64)
        occ_pred = torch.tensor(np.array([x.detach().cpu().numpy() for x in occ_grid_list]), dtype=torch.float32)
        
        ### Tensor grad required
        occ_pred = occ_pred.to(self.device).requires_grad_()
        verts_obj = verts_obj.to(self.device).requires_grad_()
        faces_obj = faces_obj.to(self.device)
        
        hand_pose = mano_pred.float().to(self.device).requires_grad_()
        hand_pose = mano.float().to(self.device)
        
        hand_pose_init = hand_pose.clone().detach()
        hand_joint_init = hand_pose_init[:, 6:].reshape(-1, 15, 3)

        mano_info_init = self.pipeline_model.forward_manolayer(hand_pose, pc_for_norm)
        mano_verts_init, mano_faces_init, mano_jtrs_init = mano_info_init['mano_verts'], mano_info_init['mano_faces'], mano_info_init['mano_jtrs']
        mano_transf = mano_info_init['mano_transf']
        
        
        verts_obj_init = verts_obj
        faces_obj_init = faces_obj
        occ_pred_init = occ_pred
        
        sensor_positions_init = self.generate_sensor_position(hand_pose_init, pc_for_norm)
        
        distance_mask_init, dist_square_init, dist_vec_init = find_points_vec_dist_mask(verts_obj_init, sensor_positions_init, self.threshold_dist)

        if "scissor" in name[0]:
            obj_name = name[0].split("-")[0]
            seq_name = name[0].split("-")[1][-3:]
            frame_name = int(name[0].split("-")[-1][-3:]) - 1
            
            obj_dir = "/mnt/public/datasets/zhenjun/VTacO_v2_obj_gt/v01"
            obj_file = os.path.join(obj_dir, "{}_{}_{:03d}.obj".format(obj_name, seq_name, frame_name))
            obj_mesh = o3d.io.read_triangle_mesh(obj_file)
            verts_diso = torch.tensor(np.array(obj_mesh.vertices), dtype=torch.float32).to(self.device)
            faces_diso = torch.tensor(np.array(obj_mesh.triangles), dtype=torch.int64).to(self.device)
            
            distance_mask_init, dist_square_init, dist_vec_init = find_points_vec_dist_mask(verts_diso.unsqueeze(0), sensor_positions_init, self.threshold_dist)
            
            verts_obj_init = verts_diso
            occ_pred_init = occ_pred
            faces_obj_init = faces_diso
                    
        
        hand_pose.requires_grad = True

        param_optim = []
        if "hand" in self.optimize_item:
            param_optim.append({"params": [hand_pose]})
        if "sdf" in self.optimize_item:
            param_optim.append({"params": [occ_pred]})
    
        mesh_name_list, mesh_hand_o3d_list, mesh_obj_o3d_list = [], [], []
        
        self.optimizer = torch.optim.Adam(param_optim, lr=self.lr)
        for e in range(self.max_steps):
            ### Define geo loss
            # hand_pose = torch.cat([hand_pose[:, :3], hand_joint], dim=1)
            loss = 0
            
            sensor_positions = self.generate_sensor_position(hand_pose, pc_for_norm)
            
            extractors = [DiffDMC(dtype=torch.float32).to(self.device) for i in range(B)]
            for bs in range(B):
                if "sdf" in self.optimize_item:
                    verts_diso, faces_diso = self.pipeline_model.extract_mesh(extractors[bs], occ_pred[bs], self.pipeline_model.padding)
                else:
                    verts_diso, faces_diso = verts_obj_init.squeeze(0), faces_obj_init.squeeze(0)
                
                    
                if "Energy" in self.loss_item.keys():
                    ### Energy term
                    E_repl, E_attr = self.compute_energy(verts_diso.unsqueeze(0), faces_diso.unsqueeze(0), sensor_positions[bs:bs+1], sensor_forces, distance_mask_init)
                    if ("sponge" in name[bs]) and ("Sequence001" in name[bs]):
                        E_attr /= 2

                    loss += self.loss_item["Energy"][0] * E_repl
                    loss += self.loss_item["Energy"][1] * E_attr
            
                
            if "hand_reg" in self.loss_item.keys():
                mano_info = self.pipeline_model.forward_manolayer(hand_pose, pc_for_norm)
                mano_verts, mano_faces, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']
                mano_transf = mano_info['mano_transf']
                
                hand_joint_squeeze = hand_pose[:, 6:].reshape(15, 3)
        
                b_axis, u_axis, l_axis = self.axis_layer(mano_jtrs, mano_transf)
                axis = hand_joint_squeeze / torch.norm(hand_joint_squeeze, dim=1, keepdim=True)
                angle = torch.norm(hand_joint_squeeze, dim=1, keepdim=False)
                # limit angle
                angle_limit_loss = HandLoss.rotation_angle_loss(angle)

                joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis)
                joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis)
                # joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis)
                
                
                loss += self.loss_item["hand_reg"][0] * angle_limit_loss
                
                if not torch.any(torch.isnan(joint_b_axis_loss)):
                    loss += self.loss_item["hand_reg"][1] * joint_b_axis_loss
                if not torch.any(torch.isnan(joint_u_axis_loss)):
                    loss += self.loss_item["hand_reg"][2] * joint_u_axis_loss
                
                # loss += self.loss_item["hand_reg"][3] * joint_l_limit_loss
            
            
            if "hand_off" in self.loss_item.keys():
                ### Hand offset loss
                loss_hand_offset = self.loss_item["hand_off"] * F.mse_loss(hand_pose, hand_pose_init)
                loss += loss_hand_offset
                
            if "obj_off" in self.loss_item.keys():
                ### Obj verts offset loss
                loss_obj_offset = self.loss_item["obj_off"] * chamfer_distance(verts_diso.unsqueeze(0), verts_obj_init)
                loss += loss_obj_offset
            
            if "sdf_off" in self.loss_item.keys():
                loss_sdf_offset = self.loss_item["sdf_off"] * F.l1_loss(occ_pred.reshape(B, -1), occ_pred_init.reshape(B, -1))
                loss += loss_sdf_offset

            if ((e+1) % self.vis_every == 0) or (e == 0):
                print("Epoch {} with loss: {:.4f}".format(e+1, loss.item()))
                # # print("Joint l limit loss:", joint_l_limit_loss.item())
                # if "Energy" in self.loss_item.keys():

                mano_info = self.pipeline_model.forward_manolayer(hand_pose, pc_for_norm)
                mano_verts, mano_faces, mano_jtrs = mano_info['mano_verts'], mano_info['mano_faces'], mano_info['mano_jtrs']
                
                
                for i in range(mano_verts.size(0)):
                    mesh_name = "{}_{:03d}".format(name[i], e+1)
                    mesh_name_list.append(mesh_name)
                    
                    mesh_hand_o3d = verts_faces_to_o3dmesh(mano_verts[i].detach().cpu().numpy(), mano_faces.cpu().numpy())
                    mesh_hand_o3d_list.append(mesh_hand_o3d)
                
                    extractor = DiffDMC(dtype=torch.float32).to(self.device)
                    verts_diso, faces_diso = self.pipeline_model.extract_mesh(extractor, occ_pred[i], self.pipeline_model.padding)
                    
                    ### For testing the distance mask
                    # distance_mask, dist_square, dist_vec = find_points_vec_dist_mask(verts_diso.unsqueeze(0), sensor_positions_init[i:i+1], self.threshold_dist)
                    verts_mask = distance_mask_init.squeeze(0).detach().cpu().numpy()
                    
                    verts_obj_i = verts_diso.detach().cpu().numpy()
                    faces_obj_i = faces_diso.detach().cpu().numpy()

                    mesh_obj_o3d = verts_faces_to_o3dmesh(verts_obj_i, faces_obj_i)
                    mesh_obj_o3d_list.append(mesh_obj_o3d)
                    
                    if verts_mask.sum() > 0:
                        verts_with_color = np.ones_like(verts_obj_i) * 255
                        for vert_id in range(verts_obj_i.shape[0]):
                            if verts_mask[vert_id].sum() > 0:
                                verts_color_id = np.argmax(verts_mask[vert_id])
                                verts_with_color[vert_id] = [10*(verts_color_id+1), 0, 0]
                                
                        
                        verts_obj_mask = np.concatenate([verts_obj_i, verts_with_color], axis=1)
                        
                        if e == 0:
                            write_ply(f"./vis/{mesh_name}_mask.ply", verts_obj_mask)
                    
            
            ### Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return mesh_name_list, mesh_obj_o3d_list, mesh_hand_o3d_list
    
    
    def generate_sensor_position(self, mano_pose, pc_for_norm):
        mano_hand_trans, mano_hand_rot = mano_pose[:, :3], mano_pose[:, 3:]
        ### Unnormed mano vertices
        mano_verts, *_ = self.pipeline_model.manolayer(mano_hand_rot)
        B, N, C = mano_verts.size()
        
        # [b,456,3,3]
        face_vertices = mano_verts[:, self.hand_flat_faces[self.hand_face_indices], :].float()
        # v2-v1，v3-v2,[b,456,2,3]
        edges = face_vertices[:, :, 1:, :] - face_vertices[:, :, :-1, :] 
        # [b,456,3]
        face_normals = torch.cross(edges[:, :, 0, :], edges[:, :, 1, :], dim=2)
        face_normals = face_normals / face_normals.norm(dim=2, keepdim=True)
        # [b,456,3]
        bary_coords_expand = self.hand_bary_coords.unsqueeze(0).expand(B, -1, -1)
        # [b,456,3]
        projected_pos = torch.einsum('bijk,bij->bik', face_vertices, bary_coords_expand)

        sensor_positions = projected_pos + self.hand_norm_off.unsqueeze(0).unsqueeze(2) * face_normals
        
        ### Transform and normalize the sensor points
        # sensor_positions += mano_hand_trans
        sensor_positions = rotate_hand_points_tensor(sensor_positions, mano_hand_trans, self.device)
        sensor_positions = norm_pointcloud_batch(sensor_positions, pc_for_norm)
    
        return sensor_positions
    
    
    def compute_energy(self, verts, faces, sensor_positions, sensor_forces, distance_mask_init):
        ### Compute Normals for the object
        normals = compute_mesh_normals_batch(verts, faces)
        
        ### Gather the forces for each region
        point_force_region = {}
        for r_name_i, r_idx_i in zip(self.region_name, self.region_idx):
            point_force_region[r_name_i] = sensor_forces[:, torch.tensor(r_idx_i)[:, 0], torch.tensor(r_idx_i)[:, 1]]

        ### K attractive string, probably computed primitively or with MLP
        if self.sensor_sample_mode == "anchor":
            k_attr = torch.stack([x.sum(dim=1) for x in point_force_region.values()], dim=1).to(self.device)

        elif self.sensor_sample_mode == "sensor":
            k_attr = torch.cat([x for x in point_force_region.values()], dim=1).to(self.device)
        
        ### Find the points that are within the threshold distance
        distance_mask, dist_square, dist_vec = find_points_vec_dist_mask(verts, sensor_positions, self.threshold_dist)
        
        ### Compute the energy
        E_repl = 0.5 * self.k_repl * torch.exp(dist_vec * normals.unsqueeze(2))[distance_mask_init].sum()
        E_attr = 0.5 * (k_attr.unsqueeze(1) * dist_square)[distance_mask_init].sum()
        
        return E_repl, E_attr