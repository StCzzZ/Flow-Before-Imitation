if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


from collections import defaultdict, deque
import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import math
from hydra.core.hydra_config import HydraConfig
from conditional_flow_matching.dataset.base_dataset import BaseDataset
from conditional_flow_matching.policy.cfm3d_shortcut import CFM3D_Shortcut
from conditional_flow_matching.policy.cfm3d import CFM3D
from conditional_flow_matching.common.pytorch_util import dict_apply, optimizer_to
from conditional_flow_matching.model.conditional_unet.ema_model import EMAModel
import sys
import requests
import cv2
import pykinect_azure as pykinect
import open3d as o3d
from typing import List

from pyrfuniverse.envs.base_env import RFUniverseBaseEnv


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainCFM3DWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    '''
    urdf_xxx: xxx that are used in isaaclab
    virtual_xxx: xxx that are used in rfuniverse
    real_xxx: xxx that are used in real world
    
    '''

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        cprint(f"Output directory: {self.output_dir}", 'magenta')
        cprint(f"learning_rate: {cfg.optimizer.lr}", 'magenta')
        self._saving_thread = None
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # configure model
        self.model: CFM3D | CFM3D_Shortcut = hydra.utils.instantiate(cfg.policy)
        self.ema_model: CFM3D | CFM3D_Shortcut = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)
        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self.stack_obs_keys = ["point_cloud", "agent_pos", "contact_forces", "hand_contact_point_all"]

        # configure training state
        self.key_warning = False
        self.obs = deque(maxlen=cfg.n_action_steps+1)
        self.n_obs_steps = cfg.n_obs_steps
        self.replay_buffer = None
        self.global_step = 0
        self.epoch = 0

        ######    Initialize the camera device    ######
        pykinect.initialize_libraries()
        # Modify camera configuration
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        print(device_config)
        self.camera_device = pykinect.start_device(config=device_config)
        self.capture = self.camera_device.update()
        self.near_plane = 450
        self.far_plane = 750


        ######     Initializing names and indices     ######
        # name sequence in rfuniverse

        virtual_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_LFJ4', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_LFJ3', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ2', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ1', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']
        real_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_THJ4', 'robot0_THJ3', 'robot0_THJ2', 'robot0_THJ1', 'robot0_THJ0', 'robot0_LFJ4', 'robot0_LFJ3', 'robot0_LFJ2', 'robot0_LFJ1', 'robot0_LFJ0', 'robot0_RFJ3', 'robot0_RFJ2', 'robot0_RFJ1', 'robot0_RFJ0', 'robot0_MFJ3', 'robot0_MFJ2', 'robot0_MFJ1', 'robot0_MFJ0', 'robot0_FFJ3', 'robot0_FFJ2', 'robot0_FFJ1', 'robot0_FFJ0']

        # name sequence in isaaclab
        urdf_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_LFJ4', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_LFJ3', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_LFJ2', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_LFJ1', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']
        actuated_joint_names = ['robot0_WRJ1', 'robot0_WRJ0', 'robot0_FFJ3', 'robot0_LFJ4', 'robot0_MFJ3', 'robot0_RFJ3', 'robot0_THJ4', 'robot0_FFJ2', 'robot0_LFJ3', 'robot0_MFJ2', 'robot0_RFJ2', 'robot0_THJ3', 'robot0_FFJ1', 'robot0_LFJ2', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_THJ2', 'robot0_FFJ0', 'robot0_LFJ1', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_THJ1', 'robot0_LFJ0', 'robot0_THJ0']
        # joints that have no force engine
        mimic_joint_names = ['robot0_FFJ0', 'robot0_MFJ0', 'robot0_RFJ0', 'robot0_LFJ0']
        to_mimic_joint_names = ['robot0_FFJ1', 'robot0_MFJ1', 'robot0_RFJ1', 'robot0_LFJ1']


        # for real_agent_pos to virtual_agent_pos
        virtual_agent_pos_joint_names = ['rh_WRJ2', 'rh_WRJ1', 'rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_LFJ5', 'rh_THJ5', 'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_LFJ4', 'rh_THJ4', 'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_LFJ3', 'rh_THJ3', 'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_LFJ2', 'rh_THJ2', 'rh_LFJ1', 'rh_THJ1']
        joint_name_url = "http://127.0.0.1:8000/joint_names/"
        wrist_name_url = "http://127.0.0.1:8000/wrist_names/"
        try:
            joint_response = requests.get(url=joint_name_url, timeout=0.5)
            real_joint_names_for_agent_pos = joint_response.json()
            wrist_response = requests.get(url=wrist_name_url, timeout=0.5)
            real_wrist_names_for_agent_pos = wrist_response.json()
            real_agent_pos_names:List = real_wrist_names_for_agent_pos + real_joint_names_for_agent_pos
        except Exception as e:
            cprint(f"joint_response: {joint_response}", "yellow")
            cprint(f"wrist_response: {wrist_response}", "yellow")
            raise RuntimeError(f"Failed to get the joint names: {e}")
        
        real_agent_pos_to_virtual_index = [real_agent_pos_names.index(name) for name in virtual_agent_pos_joint_names]
        self.real_agent_pos_to_virtual_indices = torch.tensor(real_agent_pos_to_virtual_index, dtype=torch.int32)   

        self.act_moving_average = 1.0
        self.num_crop_points = 1024
        self.action_interpolation = 0.4

        self.prev_targets = torch.zeros((1, 24), dtype=torch.float, device=self.model.device)
        self.cur_targets = torch.zeros((1, 24), dtype=torch.float, device=self.model.device)


        actuated_index = [virtual_joint_names.index(joint_name) for joint_name in actuated_joint_names]
        actuated_index.sort()
        self.actuated_dof_indices = torch.tensor(actuated_index, dtype=torch.int32, device=self.model.device)

        urdf_mimic_index = [urdf_joint_names.index(joint_name) for joint_name in mimic_joint_names]
        self.urdf_mimic_dof_indices = torch.tensor(urdf_mimic_index, dtype=torch.int32, device=self.model.device)

        urdf_to_mimic_index = [urdf_joint_names.index(joint_name) for joint_name in to_mimic_joint_names]
        self.urdf_to_mimic_dof_indices = torch.tensor(urdf_to_mimic_index, dtype=torch.int32, device=self.model.device)

        self.virtual_to_real_indices = torch.tensor([virtual_joint_names.index(joint) for joint in real_joint_names])

        urdf_to_virtual_index = [urdf_joint_names.index(joint_name) for joint_name in virtual_joint_names]
        self.urdf_to_virtual_indices = torch.tensor(urdf_to_virtual_index, dtype=torch.int32, device=self.model.device)

        virtual_to_urdf_index = [virtual_joint_names.index(joint_name) for joint_name in urdf_joint_names]
        self.virtual_to_urdf_indices = torch.tensor(virtual_to_urdf_index, dtype=torch.int32, device=self.model.device)

        self.real_agent_pos_to_virtual_indices = torch.tensor([real_agent_pos_names.index(joint) for joint in virtual_agent_pos_joint_names])

        # indices to be ajust between real and virtual
        self.THJ0_INDICE = virtual_joint_names.index("robot0_THJ0")
        self.reversed_state_indices = torch.tensor([self.THJ0_INDICE], dtype=torch.int32, device=self.model.device)

        RFU_REVERSE_NAMES = ["robot0_THJ0", "robot0_THJ1", "robot0_FFJ3", "robot0_MFJ3"]
        RFU_REVERSE_INDEX = [virtual_joint_names.index(name) for name in RFU_REVERSE_NAMES]

        self.RFU_REVERSE_INDICES = torch.tensor(RFU_REVERSE_INDEX, dtype=torch.int32, device=self.model.device)

        URDF_REVERSE_NAMES = ["robot0_THJ0", "robot0_THJ1", "robot0_FFJ3", "robot0_MFJ3"]
        # URDF_REVERSE_NAMES = ["robot0_THJ0", "robot0_FFJ3", "robot0_MFJ3"]
        URDF_REVERSE_INDEX = [virtual_joint_names.index(name) for name in URDF_REVERSE_NAMES]
        self.URDF_REVERSE_INDICES = torch.tensor(URDF_REVERSE_INDEX, dtype=torch.int32, device=self.model.device)

        self.fix_wrist = True

        self.wrist_indices = list()
        if self.fix_wrist == True:
            for wrist_name in ["robot0_WRJ1"]:   
                self.wrist_indices.append(urdf_joint_names.index(wrist_name))
            self.wrist_indices.sort()

        ########    Initializing limits ################
        # lower and upper limit for real robot, copied from the controller
        real_lower_limit = [-30.0, -45.0, -60.0, 0.0, -15.0, -30.0, 0.0, 0.0, -25.0, 0.0, 0.0, 0.0, -25.0, 0.0, 0.0, 0.0, -25.0, 0.0, 0.0, 0.0, -25.0, 0.0, 0.0, 0.0, ]
        real_upper_limit = [10.0, 35.0, 60.0, 75.0, 15.0, 30.0, 90.0, 40.0, 25.0, 90.0, 90.0, 90.0, 25.0, 90.0, 90.0, 90.0, 25.0, 90.0, 90.0, 90.0, 25.0, 90.0, 90.0, 90.0,]
        self.real_lower_tensor = torch.tensor(real_lower_limit, dtype=torch.float32, device=self.model.device)
        self.real_upper_tensor = torch.tensor(real_upper_limit, dtype=torch.float32, device=self.model.device)

        # lower and upper limit for virtual robot, copied from isaaclab(modified urdf)
        self.urdf_hand_dof_lower_limits = torch.tensor([-0.5236, -0.6981, -0.3491,  0.0000, -0.3491, -0.3491, -1.0472,  0.0000, -0.3491,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.2094,  0.0000,  0.0000,  0.0000,  0.0000, -0.5236,  0.0000,  0.0000],
       device=self.model.device)
        self.urdf_hand_dof_upper_limits = torch.tensor([0.1745, 0.4887, 0.3491, 0.7854, 0.3491, 0.3491, 1.0472, 1.5708, 0.3491, 1.5708, 1.5708, 1.2217, 1.5708, 1.5708, 1.5708, 1.5708, 0.2094, 1.5708, 1.5708, 1.5708, 1.5708, 0.5236, 1.5708, 1.5708], device=self.model.device)
        cprint(f"self.model.device: {self.model.device}", "light_red")


        self.urdf_real_lower_tensor = self.urdf_hand_dof_lower_limits[self.urdf_to_virtual_indices][self.virtual_to_real_indices]
        self.urdf_real_upper_tensor = self.urdf_hand_dof_upper_limits[self.urdf_to_virtual_indices][self.virtual_to_real_indices]


        self.hand_dof_pos = torch.zeros((1, 24), dtype=torch.float, device=self.model.device)
        self.pc_show = o3d.geometry.PointCloud()


        self.env_rfu = RFUniverseBaseEnv(check_version=False, graphics=True, log_level=0)
        self.shadow = self.env_rfu.GetAttr(456743)
        self.env_rfu.step()
        self.env_rfu.SetViewTransform(position=[0.0, 0.9, -0.15])
        self.env_rfu.ViewLookAt([0, 0.5, -0.5])
        self.env_rfu.step()

    def _pro_process_action(self, action):

        cprint(f"self.model.device: {self.model.device}", "yellow")
        cprint(f"action: {action}", "yellow")
        cprint(f"self.urdf_hand_dof_lower_limits.unsqueeze(0)[:, self.actuated_dof_indices]: {self.urdf_hand_dof_lower_limits.unsqueeze(0)[:, self.actuated_dof_indices]}", "yellow")
        action_scaled = scale(
            action,
            self.urdf_hand_dof_lower_limits.unsqueeze(0)[:, self.actuated_dof_indices],
            self.urdf_hand_dof_upper_limits.unsqueeze(0)[:, self.actuated_dof_indices],
        )
        action_scaled = self.action_interpolation * action_scaled + (1.0 - self.action_interpolation) * self.hand_dof_pos[:, self.actuated_dof_indices]
        if self.fix_wrist == True:
            for idx in self.wrist_indices:
                # action_scaled[..., idx] = self.hand_dof_pos[..., idx]    # use the current value(0 as the interpolation weight)
                action_scaled[..., idx] = 0.0    # use the current value(0 as the interpolation weight)
        return action_scaled
    

    def _proprocess_cur_target(self, cur_target):
        target_mimic = cur_target[:, self.urdf_mimic_dof_indices]
        target_to_mimic = cur_target[:, self.urdf_to_mimic_dof_indices]
        sum_mimic_to_mimic = target_mimic + target_to_mimic

        upper_to_mimic = self.urdf_hand_dof_upper_limits.unsqueeze(0)[:, self.urdf_to_mimic_dof_indices]
        over_mimic_amount = sum_mimic_to_mimic - upper_to_mimic
        over_mimic_amount = torch.clip(over_mimic_amount, 0, None)

        cur_target[:, self.urdf_mimic_dof_indices] = over_mimic_amount

        clip_sum_mimic_to_mimic = saturate(sum_mimic_to_mimic, self.urdf_hand_dof_lower_limits.unsqueeze(0)[:, self.urdf_to_mimic_dof_indices], self.urdf_hand_dof_upper_limits.unsqueeze(0)[:, self.urdf_to_mimic_dof_indices])
        cur_target[:, self.urdf_to_mimic_dof_indices] = clip_sum_mimic_to_mimic

        return cur_target


    def apply_action(self, scaled_action:torch.Tensor):
        self.cur_targets[:, self.actuated_dof_indices] = scaled_action
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.urdf_hand_dof_lower_limits.unsqueeze(0)[:, self.actuated_dof_indices],
            self.urdf_hand_dof_upper_limits.unsqueeze(0)[:, self.actuated_dof_indices],
        )
        # maybe we don't have to, cuz the real robot controller will do this
        # commented for now
        self.cur_targets[:, self.actuated_dof_indices] = self._proprocess_cur_target(self.cur_targets[:, self.actuated_dof_indices])
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def get_pc_from_camera_pyazure(self, img, depth, camera_intrinsics, camera_pose, index=None):
        if self.replay_buffer is not None:
            return self.replay_buffer['point_cloud'][index]
        else:
            # pass
            self.capture = self.camera_device.update()
            ret_color, color_image = self.capture.get_color_image()
            ret_point, points = self.capture.get_transformed_pointcloud()
            # ret_color = True
            if not ret_color or not ret_point:
                raise RuntimeError(f"Failed to get point cloud from camera, ret_color: {ret_color}, ret_point: {ret_point}")
            colors_points = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB).reshape(-1,3)/255
            points_float = points.astype(np.float64)
            colors_float = colors_points.astype(np.float64)
            points_distance = np.linalg.norm(points_float, axis=-1)
            # points_distance_tensor = torch.tensor(points_distance)

            points_filtered = points_float[(points_distance > self.near_plane) & (points_distance < self.far_plane)]
            colors_filtered = colors_float[(points_distance > self.near_plane) & (points_distance < self.far_plane)]
            cprint(f"points_distance.shape: {points_filtered.shape}", "cyan") 
            # transform, turn the points_filtered upside down
            # just for visulization, need to be adjusted if ...
            points_filtered[:, 1] = -points_filtered[:, 1]
            points_filtered[:, 2] = -points_filtered[:, 2]

            # create and draw the point cloud
            self.pc_show.points = o3d.utility.Vector3dVector(points_filtered)
            self.pc_show.colors = o3d.utility.Vector3dVector(colors_filtered)
            # do fast sampling to pc_show
            pc_show_down = o3d.geometry.PointCloud.farthest_point_down_sample(self.pc_show, self.num_crop_points)
            # o3d.visualization.draw_geometries([self.pc_show])

            points_filtered_np = np.concatenate([pc_show_down.points, pc_show_down.colors], axis=-1)
            points_filtered_tensor = torch.tensor(points_filtered_np, dtype=torch.float32, device=self.model.device)
            return points_filtered_tensor


    def get_agent_pos_from_real_world(self, index=None) -> torch.Tensor:
        if self.replay_buffer is not None:
            return self.replay_buffer['agent_pos'][index]
        else:
            joint_state_url = "http://127.0.0.1:8000/joint_status/"
            wrist_state_url = "http://127.0.0.1:8000/wrist_status/"
            try:
                joint_response = requests.get(url=joint_state_url, timeout=0.5)
                real_joint_state_tensor = torch.tensor(joint_response.json())
                wrist_response = requests.get(url=wrist_state_url, timeout=0.5)
                real_wrist_pos_tensor = torch.tensor(wrist_response.json())
                real_agent_pos_tensor = torch.cat([real_wrist_pos_tensor, real_joint_state_tensor], dim=-1)
                virtual_agent_pos_tensor = real_agent_pos_tensor[self.real_agent_pos_to_virtual_indices]
                virtual_agent_pos_angle = virtual_agent_pos_tensor / math.pi * 180.0
                virtual_agent_pos_angle = virtual_agent_pos_angle.to(self.model.device)
                virtual_agent_pos_angle[self.RFU_REVERSE_INDICES] =  - virtual_agent_pos_angle[self.RFU_REVERSE_INDICES]

                virtual_agent_pos_angle[self.URDF_REVERSE_INDICES] = - virtual_agent_pos_angle[self.URDF_REVERSE_INDICES]

                isaaclab_agent_pos_angle = virtual_agent_pos_angle[self.virtual_to_urdf_indices]

                isaaclab_agent_pos_rad = isaaclab_agent_pos_angle / 180.0 * math.pi

                self.hand_dof_pos[:] = isaaclab_agent_pos_rad.unsqueeze(0)
                cprint(f"self.dof_pos: {self.hand_dof_pos}", "green")

                isaaclab_agent_pos_unscaled = unscale(
                    isaaclab_agent_pos_rad,
                    self.urdf_hand_dof_lower_limits,
                    self.urdf_hand_dof_upper_limits,
                )
                
                return isaaclab_agent_pos_unscaled
            except Exception as e:
                print(f"{e}")
                print("Failed to get the joint state")
                return None

            

    def get_contact_data_from_real_world(self, index=None):
        if self.replay_buffer is not None:
            return self.replay_buffer['contact_forces'][index]
        else:
            cprint(f"[NotImplementedError] get_contact_data_from_real_world not implemented", 'red')
            return None
    
    def process_action_to_target_joint_state(self, urdf_action:torch.Tensor) -> torch.Tensor:
        # action(20,) -> target_joint_state(24,) 

        # just for debug
        urdf_action = torch.tensor([[-0.9996, -0.5831, -1.0022, -0.0436,  0.8660,  0.0583,  0.9218, -1.0118, 0.7587,  0.8368, -0.9139, -0.9639, -0.9971,  0.9780,  1.0198,  1.0594, 0.8633,  0.4643,  0.9582,  0.4541, -0.4841, -0.8930, -1.0724,  0.4691]], device='cuda:0')
        urdf_scaled_action = self._pro_process_action(urdf_action)



        self.apply_action(urdf_scaled_action)
        cur_targets_urdf_rad = self.cur_targets.squeeze(0)

        cur_targets_urdf_angle = cur_targets_urdf_rad * 180.0 / math.pi

        cur_targets_rfu_angle = cur_targets_urdf_angle[self.urdf_to_virtual_indices]
        cur_targets_rfu_angle[self.URDF_REVERSE_INDICES] = - cur_targets_rfu_angle[self.URDF_REVERSE_INDICES]
        self.shadow.SetJointPositionDirectly(cur_targets_rfu_angle)
        self.env_rfu.step(4)
        self.env_rfu.Pend()
        cur_targets_rfu_angle[self.RFU_REVERSE_INDICES] = - cur_targets_rfu_angle[self.RFU_REVERSE_INDICES]

        cur_targets_real_angle = cur_targets_rfu_angle[self.virtual_to_real_indices]
        
        # reverse the angles
        return cur_targets_real_angle
    
    def send_request_to_robot(self, target_state:List):
        # construct the request and send the action to real robot
        url = "http://127.0.0.1:8000/joints/"
        body = dict(positions=target_state)
        try:
            response = requests.post(url=url, json=body, timeout=0.1)
            cprint(f"response: {response}", "yellow")
        except Exception as e:
            print(f"{e}")
            print("Failed to send the control request")
    
    def reset_hand(self):
        cur_targets_pos_real = torch.zeros_like(self.cur_targets).squeeze(0)
        cur_targets_pos_list = cur_targets_pos_real.detach().cpu().tolist()
        self.send_request_to_robot(cur_targets_pos_list)
        cprint(f"resetting the hand", "yellow")
        time.sleep(5)

    def eval(self):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)
        # lastest_ckpt_path = self.get_checkpoint_path(tag="best")
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest") # changed
        cprint(f"latest_ckpt_path: {lastest_ckpt_path}", 'light_red')
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)

        # load dataset for now as no real robot data is given
        self.replay_buffer = None

        policy:CFM3D | CFM3D_Shortcut = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        cprint(f"Running real robot evaluation", 'magenta')
        # self.reset_hand()
        while True:
            # get image and depth
            img = None
            depth = None
            camera_intrinsics = None
            camera_pose = None
            obs_dict = dict()
            time_curr = time.time()
            pc = self.get_pc_from_camera_pyazure(img, depth, camera_intrinsics, camera_pose)
            cprint(f"time_used to get_pc: {time.time()-time_curr}", "yellow")

            time_curr = time.time()
            agent_pos = self.get_agent_pos_from_real_world()
            cprint(f"time_used to get_agent_pos: {time.time()-time_curr}", "yellow")
            contact_data = self.get_contact_data_from_real_world()
            obs_dict["point_cloud"] = pc
            obs_dict["agent_pos"] = agent_pos
            obs_dict["contact_forces"] = contact_data
            self.obs.append(obs_dict)

            obs_input = self._get_obs(n_steps=self.n_obs_steps)
            obs_input = dict_apply(obs_input, lambda x: x.to(policy.device, non_blocking=True))

            # add batch dimension
            obs_input['point_cloud'] = obs_input['point_cloud'].unsqueeze(0)
            obs_input['agent_pos'] = obs_input['agent_pos'].unsqueeze(0)
            for key in obs_input.keys():
                cprint(f"obs_input[{key}].shape: {obs_input[key].shape}", "yellow")
            # run the policy
            time_curr = time.time()
            with torch.no_grad():
                action_dict = policy.predict_action(obs_input)
            cprint(f"time_used to predict_action: {time.time()-time_curr}", "cyan")
            torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
            action = torch_action_dict['action'].squeeze(0)
            for act in action:

                time_curr = time.time()
                target_pos_real_tensor = self.process_action_to_target_joint_state(act)
                cprint(f"time_used to process_action_to_target_joint_state: {time.time()-time_curr}", "cyan")
                target_pos_real_list = target_pos_real_tensor.squeeze(0).detach().cpu().tolist()
                time_curr = time.time()
                cprint(f"target_pos_real_list: {target_pos_real_list}", "light_red")
                self.send_request_to_robot(target_pos_real_list)
                cprint(f"time_used to send_request_to_robot: {time.time()-time_curr}", "cyan")

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    

    def stack_last_n_obs(self, all_obs, n_steps):
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
    
    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        result = dict()
        for key in self.stack_obs_keys:
            if key not in self.obs[-1].keys() or self.obs[-1][key] is None:
                if not self.key_warning:
                    self.key_warning = True
                    cprint(f"[Warning]: Key: {key} not found in obs or obs[{key}] == None", 'red')
                continue
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs],n_steps)
        return result
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    '''
    x / () * 2 - 1, x / () 在 (0, 1)之间，所以 * 2 - 1 之后在 (-1, 1)之间
    '''
    return (2.0 * x - upper - lower) / (upper - lower)

@torch.jit.script
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

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'conditional_flow_matching', 'config'))
)
def main(cfg):
    workspace = TrainCFM3DWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
