import math
import os
import numpy as np
from pyrfuniverse.utils.coordinate_system_converter import CoordinateSystemConverter as csc
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.attributes as attr

import zarr
import numpy as np
import os
import torch
import time
import open3d as o3d
import json
from termcolor import cprint
import copy
import transforms3d as t3d
import __future__
from typing import List, Tuple, Dict
from torch import Tensor

from ball_query import query_ball_point, square_distance




def get_min_dis_and_cord(mesh_points:Tensor, point_pos:Tensor):
    square_dis = square_distance(point_pos.unsqueeze(0), mesh_points.unsqueeze(0)).squeeze(0)   # 468 * 9000
    # clip square dis to prevent negative values
    negative_indices = torch.nonzero(square_dis < 0).T
    if negative_indices.shape[-1] > 0:
        cprint(f"[Warning]: negative_indices: {negative_indices}", "yellow")  # n_indices: [316, 316]
        cprint(f"[Warning]: square_dis[negative_indices]: {square_dis[negative_indices[0], negative_indices[1]]}", "yellow")  # square_dis[negative_indices]: tensor([-2.9802e-08, -2.9802e-08], device='cuda:0')
        # raise ValueError("Negative values found in square_dis")

    square_dis = torch.clamp(square_dis, min=0)
    dis = torch.sqrt(square_dis)
    # form a 468 * 9000 * 4 tensor, [..., :3] is the coordinate of the mesh point, [3:] is the square distance
    nan_indices = torch.isnan(dis)
    if nan_indices.any():
        print("NaN values found at dis:", torch.nonzero(nan_indices))
        # check what the value is in square_dis
        print("square_dis[nan_indices]:", square_dis[nan_indices])  # square_dis[nan_indices]: tensor([-2.9802e-08, -2.9802e-08], device='cuda:0')  

    dis_coord = torch.cat([mesh_points.repeat(point_pos.shape[0], 1, 1), dis.unsqueeze(-1)], dim=-1)
    min_dis, min_indices = torch.min(dis_coord[..., 3], dim=1)
    min_square_dis_coord = dis_coord[torch.arange(dis_coord.size(0)), min_indices]
    return min_square_dis_coord[..., :3], min_square_dis_coord[..., 3]

def csc_position_transform_unity_to_isaac(csc_position:Tensor | List):
    if isinstance(csc_position, list):
        csc_position_copy = copy.deepcopy(csc_position)
        csc_position[1] = csc_position_copy[2]
        csc_position[2] = csc_position_copy[1]
        return csc_position
    elif isinstance(csc_position, Tensor):
        csc_potision_copy = copy.deepcopy(csc_position)
        csc_position[..., 1] = csc_potision_copy[..., 2]
        csc_position[..., 2] = csc_potision_copy[..., 1]
        return csc_position
    else:
        raise ValueError("csc_position should be either a list or a Tensor")


scale_dict = {
    "cube": [0.06, 0.06, 0.06],
    "vase": [0.08 * 0.01 for _ in range(3)],
    # "pyramid": [0.001, 0.001, 0.001],
    "pyramid": [0.155 * 0.01 for _ in range(3)],
    "apple": [0.16 * 0.01 for _ in range(3)],
    # "apple": [1, 1, 1],
    "A": [0.16 * 0.01 for _ in range(3)],
}

distance_threshold_dict = {
    "cube": 0.007,
    "vase": 0.007,
    "pyramid": 0.007,
    "apple": 0.008,
    "A": 0.005,
}

query_dist_dict = {
    "cube": 0.012,
    "vase": 0.012,
    "pyramid": 0.012,
    "apple": 0.012,
    "A": 0.005,
}

# pc_vis = PointcloudVisualizer()
# pc_initiated = False

#### param area #####
debug_mode = True
numpy_save_every = 500
if debug_mode:
    numpy_save_every = 1000000 # never
    pc_show = o3d.geometry.PointCloud()

obj_name = "cube"
name = "isaaclab_cube_expert_direct_openai_handinit_random_reset_singlecam_10000_rich_contact"
distance_threshold = distance_threshold_dict[obj_name]
query_radius =query_dist_dict[obj_name]
tactile_point_num_upper_bound = 456
starting_point = 0
#### end param area #####


root_dir = "/home/yijin/cfm_isaac/3D-Conditional-Flow-Matching/data/"
obj_root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl"
json_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/replay_new"
# obj_name = "cube"


zarr_path = os.path.join(root_dir, f"{name}.zarr")
zarr_save_path = os.path.join(root_dir, f"{name}_replay.zarr")
zarr_file = zarr.open(zarr_path, mode='a')

joint_state = zarr_file['data/agent_pos'][:]
joint_state = np.array(joint_state)
print(f"joint_state.shape: {joint_state.shape}")


state_data = zarr_file['data/state'][:]
state_data = torch.tensor(state_data)
obj_pos = state_data[..., 48:51].squeeze(0)
obj_rot = state_data[..., 51:55].squeeze(0)

# convert to numpy
obj_pos = obj_pos.numpy()
obj_rot = obj_rot.numpy()

# !!!!!! convert wxyz to xyzw
obj_rot = np.concatenate([obj_rot[...,1:], obj_rot[...,0:1]], axis=-1)

point_cloud_data = zarr_file['data/point_cloud'][:]

# point_dis = np.ndarray([len(joint_state), 24, 48])
print(joint_state.shape)
print(obj_pos.shape)
print(obj_rot.shape)


# open json file
with open (os.path.join(json_dir, "joint_index.json")) as json_file:
    joint_index = json.load(json_file)
joint_index = dict(joint_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initializing rfu env
if debug_mode:
    env = RFUniverseBaseEnv(check_version=False, graphics=True, log_level=0)
else:
    env = RFUniverseBaseEnv(check_version=False, graphics=False, log_level=0)

env.SetViewTransform(position=[0.0, 0.9, -0.15])
env.ViewLookAt([0, 0.5, -0.5])

env.GetViewTransform() # origin: view_pos: [0.0, 1.0, -5.0]

env.step()

view_pos = env.data['view_position']

cprint(f"view_pos: {view_pos}", "red")

# env.SetViewTransform([0, 0, 0])
# env.SendObject("LoadMesh", os.path.join(obj_root_dir, f"{obj_name}/stl.stl"))
env.SendObject("LoadMesh", os.path.join(obj_root_dir, f"{obj_name}/fbx.fbx"))
env.step()

# time_curr = time.time()

cube = env.GetAttr(654822)
cube.SetKinematic(True)
cube.SetScale(scale_dict[obj_name])
shadow = env.GetAttr(456743)
env.step()


joint_names = shadow.data["names"]
csc = csc(["right", "up", "forward"], ["right", "forward", "up"])
joint_names = ['shadow_hand', 'rootJoint', 'joints_robot0_forearm', 'robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']

names_correspondance_dict = {
    "robot0_ffdistal": "robot0_FFJ0",
    "robot0_ffmiddle": "robot0_FFJ1",
    "robot0_ffproximal": "robot0_FFJ2",
    "robot0_lfdistal": "robot0_LFJ0",
    "robot0_lfmiddle": "robot0_LFJ1",
    "robot0_lfproximal": "robot0_LFJ2",
    "robot0_mfdistal": "robot0_MFJ0",
    "robot0_mfmiddle": "robot0_MFJ1",
    "robot0_mfproximal": "robot0_MFJ2",
    "robot0_palm": "robot0_WRJ0",
    "robot0_rfdistal": "robot0_RFJ0",
    "robot0_rfmiddle": "robot0_RFJ1",
    "robot0_rfproximal": "robot0_RFJ2",
    "robot0_thdistal": "robot0_THJ0",
    "robot0_thmiddle": "robot0_THJ1",

    "robot0_FFJ0": "robot0_ffdistal",
    "robot0_FFJ1": "robot0_ffmiddle",
    "robot0_FFJ2": "robot0_ffproximal",
    "robot0_LFJ0": "robot0_lfdistal",
    "robot0_LFJ1": "robot0_lfmiddle",
    "robot0_LFJ2": "robot0_lfproximal",
    "robot0_MFJ0": "robot0_mfdistal",
    "robot0_MFJ1": "robot0_mfmiddle",
    "robot0_MFJ2": "robot0_mfproximal",
    "robot0_WRJ0": "robot0_palm",
    "robot0_RFJ0": "robot0_rfdistal",
    "robot0_RFJ1": "robot0_rfmiddle",
    "robot0_RFJ2": "robot0_rfproximal",
    "robot0_THJ0": "robot0_thdistal",
    "robot0_THJ1": "robot0_thmiddle"
}

tactile_pos_names = list(joint_index.keys())
tactile_pos_indices = list()

for key in tactile_pos_names:
    corres_name = names_correspondance_dict[key]
    indice = joint_names.index(corres_name) 
    tactile_pos_indices.append(indice)
tactile_pos_indices = torch.tensor(tactile_pos_indices)

for i in range(len(tactile_pos_names)):
    assert(tactile_pos_names[i] == names_correspondance_dict[joint_names[tactile_pos_indices[i]]])

joint2dis_indices = list()
joint2point_pos: list[Tensor] = list()

for key in tactile_pos_names:
    joint2dis_indices_sub = list()
    joint2point_pos_sub = list()
    dis_pos_list = joint_index[key]
    for tmp_lst in dis_pos_list:
        joint2dis_indices_sub.append(tmp_lst[:2])
        local_position_list = csc.cs1_pos_to_cs2_pos(tmp_lst[2:])
        local_position_list_copy = copy.deepcopy(local_position_list)
        joint2point_pos_sub.append(local_position_list)
    joint2dis_indices.append(torch.tensor(joint2dis_indices_sub, dtype=torch.int32, device=device))
    joint2point_pos.append(torch.tensor(joint2point_pos_sub, dtype=torch.float32, device=device))

joint2dis_indices_all = torch.cat(joint2dis_indices, dim=0)



lower = np.array(shadow.data["joint_lower_limit"])
upper = np.array(shadow.data["joint_upper_limit"])


hand_tactile_point_list = []
object_tactile_point_list = []
hand_object_dist_list = []
hand_object_contact_mask_list = []
point_cloud_ball_query_list = []
hand_contact_point_all_list = []

# main loop
for i in range(starting_point, len(joint_state)):
    print(f"step:{i}")
    if debug_mode:
        env.Pend()
    time_step_curr = time.time()
    cube.SetPosition(csc.cs1_pos_to_cs2_pos(obj_pos[i]))
    cube.SetRotationQuaternion(csc.cs1_quat_to_cs2_quat(obj_rot[i]))
    joint_pos = (joint_state[i] + 1) / 2 * (upper - lower) + lower
    shadow.SetJointPositionDirectly(joint_pos)
    # env.Pend()
    env.step(4)
    env.SendObject("GetPos")
    env.step()
    mesh_pos_tensor = torch.tensor(env.data["mesh_pos"], dtype = torch.float32, device=device)
    hand_point_pos_tensor = torch.tensor(env.data["point_pos"], dtype = torch.float32, device=device)
    indices = joint2dis_indices_all.T
    hand_point_pos_tensor_selected = hand_point_pos_tensor[indices[0], indices[1]]


    point_position_selected, point_distance_selected = get_min_dis_and_cord(mesh_pos_tensor, hand_point_pos_tensor_selected)

    contact_indices = torch.where(point_distance_selected < distance_threshold)[0]
    not_contact_indices = torch.where(point_distance_selected >= distance_threshold)[0]
    missing_indices = set(range(456)) - set(contact_indices.cpu().numpy()) - set(not_contact_indices.cpu().numpy())
    contact_mask = torch.zeros(point_distance_selected.shape[0], dtype=torch.int32, device=device)
    contact_mask[contact_indices] = True
    contact_indices = torch.where(contact_mask == 1)[0]

    not_contact_mask = torch.zeros(point_distance_selected.shape[0], dtype=torch.int32, device=device)

    joint_positions = torch.tensor(shadow.data["positions"], device=device)
    joint_rotations = torch.tensor(shadow.data["quaternions"], device=device)
    needed_joint_positions = joint_positions[tactile_pos_indices]
    needed_joint_rotations = joint_rotations[tactile_pos_indices]

    ######  point_cloud area ###############
    point_cloud = point_cloud_data[i]
    point_cloud = np.array(point_cloud)
    point_cloud_xyz = point_cloud[:, :3]
    point_cloud_rgb = point_cloud[:, 3:6] / 255
    point_position_world_xyz:list[Tensor] = list()
    point_position_world_rgb:list[Tensor] = list()
    mesh_point_xyz_tensor = csc_position_transform_unity_to_isaac(point_position_selected)
    mesh_point_rgb_tensor = torch.tensor([[1, 0, 1]], dtype=torch.float32, device=device).repeat(mesh_point_xyz_tensor.shape[0], 1)
    mesh_point_rgb_tensor[contact_indices] = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    point_position_world_xyz_tensor = csc_position_transform_unity_to_isaac(hand_point_pos_tensor_selected)
    point_position_world_rgb_tensor = torch.tensor([[1, 0, 1]], dtype=torch.float32, device=device).repeat(point_position_world_xyz_tensor.shape[0], 1)
    # turn the points that are contacted to green
    point_position_world_rgb_tensor[contact_indices] = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    contact_point_xyz_tensor = point_position_world_xyz_tensor[contact_indices]
    # calculate the xyz vec of the dis
    xyz_vec_dis = torch.abs(mesh_point_xyz_tensor - point_position_world_xyz_tensor)
    # do the ball query
    point_cloud_xyz_tensor = torch.tensor(point_cloud_xyz, dtype=torch.float32, device=device)
    point_cloud_rgb_tensor = torch.tensor(point_cloud_rgb, dtype=torch.float32, device=device)
    group_idx = query_ball_point(query_radius,25,point_cloud_xyz_tensor.unsqueeze(0),contact_point_xyz_tensor.unsqueeze(0))
    group_idx = group_idx.unique()
    # if point_cloud_xyz_tensor.shape[0] in group_idx, then remove it
    group_idx = group_idx[group_idx != point_cloud_xyz_tensor.shape[0]]
    group_mask = torch.zeros(point_cloud_xyz_tensor.shape[0], dtype=torch.int32, device=device)
    group_mask[group_idx] = True
    group_idx = torch.where(group_mask == 1)[0]
    cprint(f"number of contact points: {contact_indices.shape}", "green")
    cprint(f"number of queried points.shape: {group_idx.shape}", "green")
    point_cloud_rgb_tensor[group_idx] = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    point_cloud_rgb = point_cloud_rgb_tensor.cpu().numpy()

    # 收集tactile point，
    
    hand_tactile_and_ball_query = torch.cat([point_position_world_xyz_tensor[contact_indices],point_cloud_xyz_tensor[group_idx]], dim=0)
    hand_tactile_and_ball_query_xyz_ones = torch.cat([hand_tactile_and_ball_query, torch.ones(hand_tactile_and_ball_query.shape[0], 1, device=device)], dim=-1)

    hand_not_contact_xyz = point_position_world_xyz_tensor[not_contact_indices]
    hand_not_contact_xyz_shuffled = hand_not_contact_xyz[torch.randperm(hand_not_contact_xyz.size(0))]
    hand_not_conact_xyz_zeros = torch.cat([hand_not_contact_xyz_shuffled, torch.zeros(hand_not_contact_xyz_shuffled.shape[0], 1, device=device)], dim=-1)
    hand_contact_point_all = torch.cat([hand_tactile_and_ball_query_xyz_ones,hand_not_conact_xyz_zeros], dim=0)[:tactile_point_num_upper_bound]
    cprint(f"hand_contact_point_all.shape: {hand_contact_point_all.shape}", "cyan")
    if hand_contact_point_all.shape[0] < tactile_point_num_upper_bound:
        pass
    

    ###### debug mode on  ######
    if debug_mode:
        point_cloud_xyz = np.concatenate([point_cloud_xyz, point_position_world_xyz_tensor.cpu().numpy(), mesh_point_xyz_tensor.cpu().numpy()], axis=0)
        point_cloud_rgb = np.concatenate([point_cloud_rgb, point_position_world_rgb_tensor.cpu().numpy(), mesh_point_rgb_tensor.cpu().numpy()], axis=0) # 都是加在尾段,对应的

        pc_show.points = o3d.utility.Vector3dVector(point_cloud_xyz)
        pc_show.colors = o3d.utility.Vector3dVector(point_cloud_rgb)
        
        hand_contact_point_all_xyz = hand_contact_point_all[..., :3]
        hand_contact_point_all_read = hand_contact_point_all[..., 3]
        color_tensor = torch.zeros_like(hand_contact_point_all_xyz, dtype=torch.float32, device=device)
        color_tensor_indices = torch.nonzero(hand_contact_point_all_read == 1)
        if color_tensor_indices.shape[0] > 0:
            color_tensor[color_tensor_indices] = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
        pc_hand_tactile_all = o3d.geometry.PointCloud()
        cprint(f"color_tensor.shape: {color_tensor.shape}", "cyan")
        pc_hand_tactile_all.points = o3d.utility.Vector3dVector(np.asarray(hand_contact_point_all.cpu().numpy()[:, :3]))
        pc_hand_tactile_all.colors = o3d.utility.Vector3dVector(np.asarray(color_tensor.cpu().numpy()))
    ######  point_cloud area ends ###############

    ######  save area begins ###############


    hand_tactile_point_list.append(point_position_world_xyz_tensor.cpu().numpy())
    object_tactile_point_list.append(mesh_point_xyz_tensor.cpu().numpy())
    hand_object_dist_list.append(torch.cat([xyz_vec_dis, point_distance_selected.unsqueeze(-1)], dim=-1).cpu().numpy())
    hand_object_contact_mask_list.append(contact_mask.cpu().numpy())
    point_cloud_ball_query_list.append(group_mask.cpu().numpy())
    hand_contact_point_all_list.append(hand_contact_point_all.cpu().numpy())

    cprint(f"time to step: {time.time() - time_step_curr}", "cyan")
    time_step_curr = time.time()

    ###### save area ends ###################
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    if ((i % numpy_save_every) == 0 and i > 0) or (i == (len(joint_state) - 1)):
        # save the data
        cprint(f"{i} reaches time to save np")
        time_save = time.time()
        hand_tactile_numpy = np.stack(hand_tactile_point_list, axis=0)
        object_tactile_numpy = np.stack(object_tactile_point_list, axis=0)
        hand_object_dist_numpy = np.stack(hand_object_dist_list, axis=0)
        hand_object_contact_mask_numpy = np.stack(hand_object_contact_mask_list, axis=0)
        point_cloud_ball_query_numpy = np.stack(point_cloud_ball_query_list, axis=0)
        hand_contact_point_all_numpy = np.stack(hand_contact_point_all_list, axis=0)

        cprint(f"################  Record shape and Range ################", "yellow")
        cprint(f"hand_tactile_numpy: {hand_tactile_numpy.shape}, {hand_tactile_numpy.min()}, {hand_tactile_numpy.max()}", "yellow")
        cprint(f"object_tactile_numpy: {object_tactile_numpy.shape}, {object_tactile_numpy.min()}, {object_tactile_numpy.max()}", "yellow")
        cprint(f"hand_object_dist_numpy: {hand_object_dist_numpy.shape}, {hand_object_dist_numpy.min()}, {hand_object_dist_numpy.max()}", "yellow")
        cprint(f"hand_object_contact_mask_numpy: {hand_object_contact_mask_numpy.shape}, {hand_object_contact_mask_numpy.min()}, {hand_object_contact_mask_numpy.max()}", "yellow")
        cprint(f"point_cloud_ball_query_numpy: {point_cloud_ball_query_numpy.shape}, {point_cloud_ball_query_numpy.min()}, {point_cloud_ball_query_numpy.max()}", "yellow")
        cprint(f"hand_contact_point_all_numpy: {hand_contact_point_all_numpy.shape}, {hand_contact_point_all_numpy.min()}, {hand_contact_point_all_numpy.max()}", "yellow")
        cprint(f"################  Record shape and Range ################", "yellow")

        if i == numpy_save_every:
            zarr_hand_tactile_point = zarr.array(hand_tactile_numpy, chunks=(2000, hand_tactile_numpy.shape[1], hand_tactile_numpy.shape[2]), compressor=compressor)
            zarr_file['data/hand_tactile_point'] = zarr_hand_tactile_point
            hand_tactile_point_list.clear()

            zarr_object_tactile_point = zarr.array(object_tactile_numpy, chunks=(2000, object_tactile_numpy.shape[1], object_tactile_numpy.shape[2]), compressor=compressor)
            zarr_file['data/object_tactile_point'] = zarr_object_tactile_point
            object_tactile_point_list.clear()

            zarr_hand_object_dist = zarr.array(hand_object_dist_numpy, chunks=(2000, hand_object_dist_numpy.shape[1]), compressor=compressor)
            zarr_file['data/hand_object_dist'] = zarr_hand_object_dist
            hand_object_dist_list.clear()

            zarr_hand_object_contact_mask = zarr.array(hand_object_contact_mask_numpy, chunks=(2000, hand_object_contact_mask_numpy.shape[1]), compressor=compressor)
            zarr_file['data/hand_object_contact_mask'] = zarr_hand_object_contact_mask
            hand_object_contact_mask_list.clear()

            zarr_point_cloud_ball_query = zarr.array(point_cloud_ball_query_numpy, chunks=(2000, point_cloud_ball_query_numpy.shape[1]), compressor=compressor)
            zarr_file['data/point_cloud_ball_query_mask'] = zarr_point_cloud_ball_query
            point_cloud_ball_query_list.clear()

            zarr_hand_contact_point_all = zarr.array(hand_contact_point_all_numpy, chunks=(2000, hand_contact_point_all_numpy.shape[1], hand_contact_point_all_numpy.shape[2]), compressor=compressor)
            zarr_file['data/hand_contact_point_all'] = zarr_hand_contact_point_all
            hand_contact_point_all_list.clear()


        else:
            zarr_hand_tactile_point = zarr.array(hand_tactile_numpy, chunks=(2000, hand_tactile_numpy.shape[1], hand_tactile_numpy.shape[2]), compressor=compressor)
            zarr_file['data/hand_tactile_point'].append(zarr_hand_tactile_point)
            hand_tactile_point_list.clear()

            zarr_object_tactile_point = zarr.array(object_tactile_numpy, chunks=(2000, object_tactile_numpy.shape[1], object_tactile_numpy.shape[2]), compressor=compressor)
            zarr_file['data/object_tactile_point'].append(zarr_object_tactile_point)
            object_tactile_point_list.clear()

            zarr_hand_object_dist = zarr.array(hand_object_dist_numpy, chunks=(2000, hand_object_dist_numpy.shape[1]), compressor=compressor)
            zarr_file['data/hand_object_dist'].append(zarr_hand_object_dist)
            hand_object_dist_list.clear()

            zarr_hand_object_contact_mask = zarr.array(hand_object_contact_mask_numpy, chunks=(2000, hand_object_contact_mask_numpy.shape[1]), compressor=compressor)
            zarr_file['data/hand_object_contact_mask'].append(zarr_hand_object_contact_mask)
            hand_object_contact_mask_list.clear()

            zarr_point_cloud_ball_query = zarr.array(point_cloud_ball_query_numpy, chunks=(2000, point_cloud_ball_query_numpy.shape[1]), compressor=compressor)
            zarr_file['data/point_cloud_ball_query_mask'].append(zarr_point_cloud_ball_query)
            point_cloud_ball_query_list.clear()

            zarr_hand_contact_point_all = zarr.array(hand_contact_point_all_numpy, chunks=(2000, hand_contact_point_all_numpy.shape[1], hand_contact_point_all_numpy.shape[2]), compressor=compressor)
            zarr_file['data/hand_contact_point_all'].append(zarr_hand_contact_point_all)
            hand_contact_point_all_list.clear()
        
        cprint(f"time to save: {time.time() - time_save}", "cyan")

        # # save zarr file
        # zarr.save(zarr_save_path, zarr_file)



    

    