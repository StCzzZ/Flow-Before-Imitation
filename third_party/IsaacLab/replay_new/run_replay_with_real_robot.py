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
import requests

from ball_query import query_ball_point, square_distance

# import os
# os.environ.pop("http_proxy", None)
# os.environ.pop("https_proxy", None)   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape is (N, dims) or (dims,).
        upper: The maximum value of the tensor. Shape is (N, dims) or (dims,).

    Returns:
        Clamped transform of the tensor. Shape is (N, dims).
    """
    return torch.max(torch.min(x, upper), lower)

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
# name = "isaaclab_cube_expert_direct_openai_handinit_random_reset_singlecam_10000_rich_contact"
name = "isaaclab_cube_real_expertinhand_reorient_cube_real_decimation10"
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
zarr_file = zarr.open(zarr_path, mode='r')

joint_state = zarr_file['data/agent_pos'][:]
joint_state = np.array(joint_state)
print(f"joint_state.shape: {joint_state.shape}")


state_data = zarr_file['data/state'][:]
state_data = torch.tensor(state_data)
obj_pos = state_data[..., 48:51].squeeze(0)
obj_rot = state_data[..., 51:55].squeeze(0)

action_data = zarr_file['data/action'][:]
action_data = torch.tensor(action_data)

pc_data = zarr_file['data/point_cloud'][:]

episode_end_data = zarr_file['meta/episode_ends'][:]
cprint(f"episode_end_data: {episode_end_data}", "yellow")




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
obj_name = "cube"
env.SendObject("LoadMesh", os.path.join(obj_root_dir, f"{obj_name}/fbx.fbx"))
env.step()
# cube = env.GetAttr(654822)
# cube.SetKinematic(True)
# cube.SetScale(scale_dict[obj_name])
shadow = env.GetAttr(456743)
env.step()
joint_names = shadow.data["names"]
csc = csc(["right", "up", "forward"], ["right", "forward", "up"])
joint_names = ['shadow_hand', 'rootJoint', 'joints_robot0_forearm', 'robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']

# for sending action to robot
urdf_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_LFJ4', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_LFJ3', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_LFJ2', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_LFJ1', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']

virtual_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']
actuated_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_THJ0']
mimic_joint_names = ['robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ0']
force_same_joint_names = ['robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ1']
real_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_THJ4', 'robot0_THJ3', 'robot0_THJ2', 'robot0_THJ1', 'robot0_THJ0', 'robot0_LFJ4', 'robot0_LFJ3', 'robot0_LFJ2', 'robot0_LFJ1', 'robot0_LFJ0', 'robot0_RFJ3', 'robot0_RFJ2', 'robot0_RFJ1', 'robot0_RFJ0', 'robot0_MFJ3', 'robot0_MFJ2', 'robot0_MFJ1', 'robot0_MFJ0', 'robot0_FFJ3', 'robot0_FFJ2', 'robot0_FFJ1', 'robot0_FFJ0']
actuated_index = [virtual_joint_names.index(joint_name) for joint_name in actuated_joint_names]
actuated_index.sort()
actuated_dof_indices = torch.tensor(actuated_index, dtype=torch.int32, device=device)
cprint(f"actuated_dof_indices: {actuated_dof_indices}", "green")
mimic_index = [virtual_joint_names.index(joint_name) for joint_name in mimic_joint_names]
mimic_dof_indices = torch.tensor(mimic_index, dtype=torch.int32, device=device)
force_same_index = [virtual_joint_names.index(joint_name) for joint_name in force_same_joint_names]
force_same_dof_indices = torch.tensor(force_same_index, dtype=torch.int32, device=device)

virtual_to_real_index = [virtual_joint_names.index(joint_name) for joint_name in real_joint_names]
virtual_to_real_indices = torch.tensor(virtual_to_real_index, dtype=torch.int32, device=device)

urdf_to_virtual_index = [urdf_joint_names.index(joint_name) for joint_name in virtual_joint_names]
urdf_to_virtual_indices = torch.tensor(urdf_to_virtual_index, dtype=torch.int32, device=device)


lower = np.array(shadow.data["joint_lower_limit"])
upper = np.array(shadow.data["joint_upper_limit"])


urdf_hand_dof_lower_limits = torch.tensor([-0.5236, -0.6981, -0.3491,  0.0000, -0.3491, -0.3491, -1.0472,  0.0000, -0.3491,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.2094,  0.0000,  0.0000,  0.0000,  0.0000, -0.5236,  0.0000,  0.0000],
       device=device)
urdf_and_dof_upper_limits = torch.tensor([0.1745, 0.4887, 0.3491, 0.7854, 0.3491, 0.3491, 1.0472, 1.5708, 0.3491,
         1.5708, 1.5708, 1.2217, 1.5708, 1.5708, 1.5708, 1.5708, 0.2094, 1.5708,
         1.5708, 1.5708, 1.5708, 0.6981, 1.5708, 1.5708], device=device)




real_lower_limit = [
    -30.0,
    -45.0,
    -60.0,
    0.0,
    -15.0,
    -30.0,
    0.0,
    0.0,
    -25.0,
    0.0,
    0.0,
    0.0,
    -25.0,
    0.0,
    0.0,
    0.0,
    -25.0,
    0.0,
    0.0,
    0.0,
    -25.0,
    0.0,
    0.0,
    0.0,
]
real_upper_limit = [
    10.0,
    35.0,
    60.0,
    75.0,
    15.0,
    30.0,
    90.0,
    40.0,
    25.0,
    90.0,
    90.0,
    90.0,
    25.0,
    90.0,
    90.0,
    90.0,
    25.0,
    90.0,
    90.0,
    90.0,
    25.0,
    90.0,
    90.0,
    90.0,
]


real_lower_tensor = torch.tensor(real_lower_limit, dtype=torch.float32, device=device)
real_upper_tensor = torch.tensor(real_upper_limit, dtype=torch.float32, device=device)

lower_tensor = torch.tensor(lower, dtype=torch.float32, device=device)
upper_tensor = torch.tensor(upper, dtype=torch.float32, device=device)

lower_tensor = lower_tensor[virtual_to_real_indices]
upper_tensor = upper_tensor[virtual_to_real_indices]

cprint(f"lower_tensor: {lower_tensor}", "green")
cprint(f"real_lower_tensor: {real_lower_tensor}", "green")

cprint(f"upper_tensor: {upper_tensor}", "magenta")
cprint(f"real_upper_tensor: {real_upper_tensor}", "magenta")

URDF_REVERSE_NAMES = ["robot0_THJ0", "robot0_THJ1", "robot0_FFJ3", "robot0_MFJ3"]
# URDF_REVERSE_NAMES = ["robot0_THJ0", "robot0_FFJ3", "robot0_MFJ3"]
URDF_INDEX = [virtual_joint_names.index(name) for name in URDF_REVERSE_NAMES]
URDF_INDICES = torch.tensor(URDF_INDEX, dtype=torch.int32, device=device)

THJ0_INDICE = real_joint_names.index("robot0_THJ0")
THJ0_INDICES = torch.tensor([THJ0_INDICE], dtype=torch.int32, device=device)

REAL_REVERSE_NAMES = ["robot0_THJ1", "robot0_FFJ3", "robot0_MFJ3"]
REAL_INDEX = [real_joint_names.index(name) for name in REAL_REVERSE_NAMES]
REAL_INDICES = torch.tensor(REAL_INDEX, dtype=torch.int32, device=device)

# just for debug
# THJ0_INDICES = torch.tensor([], dtype=torch.int32, device=device)
cprint(f"THJ0_INDICES: {THJ0_INDICES}", "green")

urdf_real_lower_tensor = urdf_hand_dof_lower_limits[urdf_to_virtual_indices][virtual_to_real_indices]
urdf_real_upper_tensor = urdf_and_dof_upper_limits[urdf_to_virtual_indices][virtual_to_real_indices]

cprint(f"urdf_real_lower_tensor: {urdf_real_lower_tensor}", "grey")
cprint(f"urdf_real_upper_tensor: {urdf_real_upper_tensor}", "grey")


act_moving_average = 0.3
hand_dof_lower_limits = torch.tensor([[-0.4890, -0.6980, -0.3490, -0.3490, -0.3490,  0.0000, -1.0470,  0.0000,
          0.0000,  0.0000, -0.3490,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         -0.2090,  0.0000,  0.0000,  0.0000,  0.0000, -0.5240,  0.0000, -1.5710]],
       device=device)
hand_dof_upper_limits = torch.tensor([[0.1400, 0.4890, 0.3490, 0.3490, 0.3490, 0.7850, 1.0470, 1.5710, 1.5710,
         1.5710, 0.3490, 1.2220, 1.5710, 1.5710, 1.5710, 1.5710, 0.2090, 1.5710,
         1.5710, 1.5710, 1.5710, 0.5240, 1.5710, 0.0000]], device=device)
prev_targets = torch.zeros((1, 24), dtype=torch.float, device=device)
cur_targets = torch.zeros((1, 24), dtype=torch.float, device=device)

# a isaaclab_action to a isaaclab state

def apply_action(act:torch.Tensor):
    cur_targets[:, actuated_dof_indices] = scale(
        act,
        hand_dof_lower_limits[:, actuated_dof_indices],
        hand_dof_upper_limits[:, actuated_dof_indices],
    )
    cur_targets[:, actuated_dof_indices] = (
        act_moving_average * cur_targets[:, actuated_dof_indices]
        + (1.0 - act_moving_average) * prev_targets[:, actuated_dof_indices]
    )
    
    cur_targets[:, actuated_dof_indices] = saturate(
        cur_targets[:, actuated_dof_indices],
        hand_dof_lower_limits[:, actuated_dof_indices],
        hand_dof_upper_limits[:, actuated_dof_indices],
    )

    cprint(f"cur_targets: {cur_targets}", "yellow")
    cur_targets[:, mimic_dof_indices] = cur_targets[:, force_same_dof_indices]
    cprint(f"cur_targets after: {cur_targets}", "cyan")

    cur_targets[:, mimic_dof_indices] = saturate(
        cur_targets[:, mimic_dof_indices],
        hand_dof_lower_limits[:, mimic_dof_indices],
        hand_dof_upper_limits[:, mimic_dof_indices],
    )

    cprint(f"cur_targets after saturate: {cur_targets}", "green")

    
    prev_targets[:] = cur_targets[:]
    return cur_targets


# action_count = 0

# # main loop
# # for i in range(starting_point, len(joint_state)):
for i in range(starting_point, episode_end_data[0]):
    print(f"step:{i}")
    cprint(f"joint_state[i]: {joint_state[i]}", "green")
    action_state_debug = action_data[i]
    joint_state_debug = joint_state[i]
    

    joint_state_debug = torch.tensor(joint_state_debug, dtype=torch.float32, device=device)
    action_state_debug = torch.tensor(action_state_debug, dtype=torch.float32, device=device)
    cprint(f"{'real' in name}", "cyan")
    if "real" in name:
        joint_state_debug = joint_state_debug[urdf_to_virtual_indices]
        joint_state_debug[URDF_INDICES] = -joint_state_debug[URDF_INDICES]

        action_state_debug = action_state_debug[urdf_to_virtual_indices]
        action_state_debug[URDF_INDICES] = -action_state_debug[URDF_INDICES]

    joint_state_debug = joint_state_debug.detach().cpu().numpy()
    action_state_debug = action_state_debug.detach().cpu().numpy()
    if i == 0:
        action_state_debug = joint_state_debug
    cprint(f"joint_state_debug: {joint_state_debug}", "cyan")
    # JUST FOR DEBUG
    joint_pos = (joint_state_debug + 1) / 2 * (upper - lower) + lower
    action_pos = (action_state_debug + 1) / 2 * (upper - lower) + lower

    cprint(f"joint_pos: {joint_pos}", "red")

    time_step_curr = time.time()

    if "real" in name:
        shadow.SetPosition([0.0, 0.5, 0.055])
    shadow.SetJointPositionDirectly(joint_pos)
    # shadow.SetJointPositionDirectly(action_pos)
    # cube.SetPosition(csc.cs1_pos_to_cs2_pos(obj_pos[i]))
    # cube.SetRotationQuaternion(csc.cs1_quat_to_cs2_quat(obj_rot[i]))
    env.step(4)


    joint_state_tensor = torch.tensor(joint_state[i], dtype=torch.float32, device=device)
    action_state_tensor = torch.tensor(action_data[i], dtype=torch.float32, device=device)
    if "real" in name:
        joint_state_tensor = torch.tensor(joint_state_debug, dtype=torch.float32, device=device)
        action_state_tensor = torch.tensor(action_state_debug, dtype=torch.float32, device=device)
    # convert to real robot sequence
    joint_state_tensor_real = joint_state_tensor[virtual_to_real_indices]
    action_state_tensor_real = action_state_tensor[virtual_to_real_indices]
    cprint(f"joint_state_tensor_real: {joint_state_tensor_real}", "magenta")
    # convert joint state to real format
    joint_pos_tensor = (joint_state_tensor_real + 1) / 2 * (real_upper_tensor - real_lower_tensor) + real_lower_tensor
    action_pos_tensor = (action_state_tensor_real + 1) / 2 * (real_upper_tensor - real_lower_tensor) + real_lower_tensor
    # reverse some joints to meet real robot protocol

    joint_pos_tensor[THJ0_INDICES] = (joint_state_tensor_real[THJ0_INDICES] + 1) / 2 * (real_lower_tensor[THJ0_INDICES] - real_upper_tensor[THJ0_INDICES]) + real_upper_tensor[THJ0_INDICES]

    action_pos_tensor[THJ0_INDICES] = (action_state_tensor_real[THJ0_INDICES] + 1) / 2 * (real_lower_tensor[THJ0_INDICES] - real_upper_tensor[THJ0_INDICES]) + real_upper_tensor[THJ0_INDICES]

    joint_pos_tensor[REAL_INDICES] = -joint_pos_tensor[REAL_INDICES]
    action_pos_tensor[REAL_INDICES] = -action_pos_tensor[REAL_INDICES]

    cprint(f"real_joint_pos_tensor[THJ0_INDICES]: {joint_pos_tensor[THJ0_INDICES]}", "green")

    joint_pos_list = joint_pos_tensor.detach().cpu().tolist()
    action_pos_list = action_pos_tensor.detach().cpu().tolist()

    # calculate current state from actions
    
    cprint(f"joint_pos_list: {joint_pos_list}", "green")

    cprint(f"*" * 50, "grey")

    # send joint_state to real robot
    # construct the request
    url = "http://127.0.0.1:8000/joints/"
    # url = "http://127.0.0.1:8000/rh_trajectory_controller/state"
    body = dict(positions=action_pos_list)
    try:
        response = requests.post(url=url, json=body, timeout=0.5)

        cprint(f"Success sending: {response}", "green")
        time.sleep(0.01)
        if i == 0:
            time.sleep(5)
            env.Pend()
    except Exception as e:
        print(f"{e}")
        print("Failed to send the control request")
    time.sleep(1 / 3)
    # env.Pend()
    # exit(0)
    
env.Pend()
    



    

    