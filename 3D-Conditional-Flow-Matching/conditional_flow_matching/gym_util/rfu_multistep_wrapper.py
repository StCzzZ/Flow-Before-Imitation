import gym
import gym.spaces
import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from collections import defaultdict, deque
import dill


import os
import json
import copy
import transforms3d as t3d
import __future__
from typing import List, Tuple, Dict
from torch import Tensor
from pyrfuniverse.utils.coordinate_system_converter import CoordinateSystemConverter as csc
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
import pyrfuniverse.attributes as attr
from termcolor import cprint


# gloabal variables
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


def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return gym.spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, gym.spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, gym.spaces.Dict):
        result_space = gym.spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x[-n:])
    else:
        return np.array(x[-n:])



def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method='max'):
    if isinstance(data[0], torch.Tensor):
        if method == 'max':
            # equivalent to any
            return torch.max(torch.stack(data))
        elif method == 'min':
            # equivalent to all
            return torch.min(torch.stack(data))
        elif method == 'mean':
            return torch.mean(torch.stack(data))
        elif method == 'sum':
            return torch.sum(torch.stack(data))
        else:
            raise NotImplementedError()
    else:
        if method == 'max':
            # equivalent to any
            return np.max(data)
        elif method == 'min':
            # equivalent to all
            return np.min(data)
        elif method == 'mean':
            return np.mean(data)
        elif method == 'sum':
            return np.sum(data)
        else:
            raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape, 
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class RFUMultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max',
            object_name = 'cube', 
            device = torch.device("cuda:0"),
            vec_dim = 3,
        ):

        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.all_keys = ["hand_contact_point_all"]

        self.obs = deque(maxlen=n_obs_steps+1)
        self.device = device
        self.vec_dim = vec_dim


        self.scale_dict = {
            "cube": [0.06, 0.06, 0.06],
            "vase": [0.08 * 0.01 for _ in range(3)],
            "pyramid": [0.155 * 0.01 for _ in range(3)],
            "apple": [0.16 * 0.01 for _ in range(3)],
            "A": [0.16 * 0.01 for _ in range(3)],
        }
        self.distance_threshold_dict = {
            "cube": 0.007,
            "vase": 0.007,
            "pyramid": 0.007,
            "apple": 0.008,
            "A": 0.005,
        }
        self.query_dist_dict = {
            "cube": 0.012,
            "vase": 0.012,
            "pyramid": 0.012,
            "apple": 0.012,
            "A": 0.005,
        }
        self.tactile_point_num_upper_bound = 456
        self.object_name = object_name
        self.obj_root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl"
        self.json_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/replay"

        with open (os.path.join(self.json_dir, "joint_index.json")) as json_file:
            joint_index = json.load(json_file)
        self.joint_index = dict(joint_index)

        self.env_rfu = RFUniverseBaseEnv(check_version=False, graphics=False, log_level=0)
        # self.env_rfu = RFUniverseBaseEnv(check_version=False, graphics=True, log_level=0)

        self.env_rfu.SetViewTransform(position=[0.0, 0.9, -0.15])
        self.env_rfu.ViewLookAt([0, 0.5, -0.5])
        self.env_rfu.SendObject("LoadMesh", os.path.join(self.obj_root_dir, f"{self.object_name}/fbx.fbx"))
        self.env_rfu.step()

        print(f"object_path: {os.path.join(self.obj_root_dir, f'{self.object_name}/fbx.fbx')}")

        self.object_rfu = self.env_rfu.GetAttr(654822)
        self.object_rfu.SetKinematic(True)
        self.object_rfu.SetScale(self.scale_dict[self.object_name])
        self.shadow = self.env_rfu.GetAttr(456743)
        self.joint_names = self.shadow.data["names"]
        self.csc_rfu = csc(["right", "up", "forward"], ["right", "forward", "up"])


        self.tactile_pos_names = list(joint_index.keys())
        tactile_pos_indices = list()
        for key in self.tactile_pos_names:
            corres_name = names_correspondance_dict[key]
            indice = joint_names.index(corres_name) # 表示该joint_name 在joint_position里面是第几个
            tactile_pos_indices.append(indice)
        self.tactile_pos_indices = torch.tensor(tactile_pos_indices)
        for i in range(len(self.tactile_pos_names)):
            assert(self.tactile_pos_names[i] == names_correspondance_dict[joint_names[tactile_pos_indices[i]]])
        # 接下来的排序都以tactile_pos_names[i]为准好了
        # 这两个list 的顺序是按照names[i]的顺序来的，里面的pos和indice也是一一对应
        joint2dis_indices = list()
        joint2point_pos: list[Tensor] = list()

        for key in self.tactile_pos_names:
            joint2dis_indices_sub = list()
            joint2point_pos_sub = list()
            dis_pos_list = joint_index[key]
            for tmp_lst in dis_pos_list:
                joint2dis_indices_sub.append(tmp_lst[:2])
                local_position_list = self.csc_rfu.cs1_pos_to_cs2_pos(tmp_lst[2:])
                joint2point_pos_sub.append(local_position_list)
            joint2dis_indices.append(torch.tensor(joint2dis_indices_sub, dtype=torch.int64, device=device))
            joint2point_pos.append(torch.tensor(joint2point_pos_sub, dtype=torch.float32, device=device))

        # 这里多append了一次
        # joint2dis_indices.append(torch.tensor(joint2dis_indices_sub, dtype=torch.int64, device=device))
        # joint2point_pos.append(torch.tensor(joint2point_pos_sub, dtype=torch.float32, device=device))

        self.joint2point_pos = joint2point_pos
        self.joint2dis_indices_all = torch.cat(joint2dis_indices, dim=0)
        self.distance_threshold = self.distance_threshold_dict[self.object_name]
        self.query_radius = self.query_dist_dict[self.object_name]
        self.env_rfu.step()
        self.shadow_lower_rfu = np.array(self.shadow.data["joint_lower_limit"])
        self.shadow_upper_rfu = np.array(self.shadow.data["joint_upper_limit"])
    

    def step(self, obj_pos_arr, obj_rot_arr, joint_state_arr, point_cloud_arr):

        observation = dict()
        joint_state_arr = joint_state_arr.cpu().numpy()
        obj_pos_arr = obj_pos_arr.cpu().numpy()
        obj_rot_arr = obj_rot_arr.cpu().numpy()

        """
        actions: (n_action_steps,) + action_shape
        """
        # assert(obj_pos_arr.shape[0] == self.n_action_steps)
        # assert(obj_rot_arr.shape[0] == self.n_action_steps)
        # assert(joint_state_arr.shape[0] == self.n_action_steps)
        for i in range(joint_state_arr.shape[0]):
            

            if isinstance(point_cloud_arr, np.ndarray):
                point_cloud = torch.tensor(point_cloud_arr[i], dtype=torch.float32, device=self.device)
            else:
                point_cloud = point_cloud_arr[i].to(dtype=torch.float32, device=self.device)

            point_cloud_xyz_tensor = point_cloud[..., :3]
            point_cloud_rgb_tensor = point_cloud[..., 3:]

            obj_rot = np.concatenate([obj_rot_arr[i][...,1:], obj_rot_arr[i][...,0:1]], axis=-1)

            self.object_rfu.SetPosition(self.csc_rfu.cs1_pos_to_cs2_pos(obj_pos_arr[i]))
            self.object_rfu.SetRotationQuaternion(self.csc_rfu.cs1_quat_to_cs2_quat(obj_rot))

            joint_pos = (joint_state_arr[i] + 1) / 2 * (self.shadow_upper_rfu - self.shadow_lower_rfu) + self.shadow_lower_rfu
            self.shadow.SetJointPositionDirectly(joint_pos)
            self.env_rfu.step(4)
            # self.env_rfu.SendObject("GetDis")
            self.env_rfu.SendObject("GetPos")
            self.env_rfu.step()
            mesh_pos_tensor = torch.tensor(self.env_rfu.data["mesh_pos"], dtype = torch.float32, device=self.device)
            hand_point_pos_tensor = torch.tensor(self.env_rfu.data["point_pos"], dtype = torch.float32, device=self.device)
            indices = self.joint2dis_indices_all.T
            hand_point_pos_tensor_selected = hand_point_pos_tensor[indices[0], indices[1]]
            point_position_selected, point_distance_selected = self.get_min_dis_and_cord(mesh_pos_tensor, hand_point_pos_tensor_selected)
            contact_indices = torch.where(point_distance_selected < self.distance_threshold)[0]

            contact_indices = torch.where(point_distance_selected < self.distance_threshold)[0]
            not_contact_indices = torch.where(point_distance_selected >= self.distance_threshold)[0]
            
            joint_positions = torch.tensor(self.shadow.data["positions"], device=self.device)
            joint_rotations = torch.tensor(self.shadow.data["quaternions"], device=self.device)

            needed_joint_positions = joint_positions[self.tactile_pos_indices]
            needed_joint_rotations = joint_rotations[self.tactile_pos_indices]

            mesh_point_xyz_tensor = self.csc_position_transform_unity_to_isaac(point_position_selected)
            point_position_world_xyz_tensor = self.csc_position_transform_unity_to_isaac(hand_point_pos_tensor_selected)

            contact_point_xyz_tensor = point_position_world_xyz_tensor[contact_indices]
            xyz_vec_dis = torch.abs(mesh_point_xyz_tensor - point_position_world_xyz_tensor)
            group_idx = self.query_ball_point(self.query_radius,25,point_cloud_xyz_tensor.unsqueeze(0),contact_point_xyz_tensor.unsqueeze(0))
            group_idx = group_idx.unique()
            group_idx = group_idx[group_idx != point_cloud_xyz_tensor.shape[0]]
            #TODO not done

            point_position_world_xyz_vec_tensor = torch.zeros((point_position_world_xyz_tensor.shape[0], point_position_world_xyz_tensor.shape[1] + self.vec_dim), device=self.device)

            point_position_world_xyz_vec_tensor[:, :point_position_world_xyz_tensor.shape[1]] = point_position_world_xyz_tensor

            # 收集tactile point，
    
            hand_tactile_and_ball_query = torch.cat([point_position_world_xyz_tensor[contact_indices],point_cloud_xyz_tensor[group_idx]], dim=0)
            # hand_tactile_and_ball_query = torch.cat([point_position_world_xyz_tensor[contact_indices]], dim=0)
            hand_tactile_and_ball_query_xyz_ones = torch.cat([hand_tactile_and_ball_query, torch.ones(hand_tactile_and_ball_query.shape[0], 1, device=self.device)], dim=-1)


            hand_not_contact_xyz = point_position_world_xyz_tensor[not_contact_indices]
            hand_not_contact_xyz_shuffled = hand_not_contact_xyz[torch.randperm(hand_not_contact_xyz.size(0))]
            hand_not_conact_xyz_zeros = torch.cat([hand_not_contact_xyz_shuffled, torch.zeros(hand_not_contact_xyz_shuffled.shape[0], 1, device=self.device)], dim=-1)

            hand_contact_point_all = torch.cat([hand_tactile_and_ball_query_xyz_ones,hand_not_conact_xyz_zeros], dim=0)[:self.tactile_point_num_upper_bound]
            # hand_contact_point_all = torch.cat([hand_tactile_and_ball_query_xyz_ones], dim=0)[:self.tactile_point_num_upper_bound]

            observation["hand_contact_point_all"] = hand_contact_point_all

            self.obs.append(observation)

        observation = self._get_obs(self.n_obs_steps)
        # self.env_rfu.Pend()
        return observation

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)

        result = dict()
        for key in self.all_keys:
            result[key] = stack_last_n_obs(
                [obs[key] for obs in self.obs],
                n_steps
            )
        return result

    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.

        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def get_min_dis_and_cord(self, mesh_points:Tensor, point_pos:Tensor):
        square_dis = self.square_distance(point_pos.unsqueeze(0), mesh_points.unsqueeze(0)).squeeze(0)   # 468 * 9000
        dis = torch.sqrt(square_dis)
        # form a 468 * 9000 * 4 tensor, [..., :3] is the coordinate of the mesh point, [3:] is the square distance
        dis_coord = torch.cat([mesh_points.repeat(point_pos.shape[0], 1, 1), dis.unsqueeze(-1)], dim=-1)
        min_dis, min_indices = torch.min(dis_coord[..., 3], dim=1)
        min_square_dis_coord = dis_coord[torch.arange(dis_coord.size(0)), min_indices]
        return min_square_dis_coord[..., :3], min_square_dis_coord[..., 3]

    def csc_position_transform_unity_to_isaac(self, csc_position:Tensor | List):
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
        
    def query_ball_point(self, radius, nsample, xyz, new_xyz):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx
