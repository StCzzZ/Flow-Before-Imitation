# add isaaclab into syspath
# import os
# import sys
# from termcolor import  cprint
# IsaacLab_path = os.path.join(os.path.abspath(os.path.curdir), "third_party/IsaacLab")
# # cprint(f"IsaacLab_path: {IsaacLab_path}", "cyan")
# sys.path.append(IsaacLab_path)



# Regular Imports
import gymnasium as gym
import math
import torch
from termcolor import cprint
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import argparse
import tqdm
import wandb
# import pytorch3d

# from rl_games
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

# import from omni app
from omni.isaac.lab.app import AppLauncher


# import from cfm
from conditional_flow_matching.env_runner.base_runner import BaseRunner
from conditional_flow_matching.policy.base_policy import BasePolicy
from conditional_flow_matching.common.pytorch_util import dict_apply
import conditional_flow_matching.common.logger_util as logger_util
from conditional_flow_matching.gym_util.isaac_multistep_wrapper import MultiStepWrapper
from conditional_flow_matching.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from conditional_flow_matching.common.replay_buffer import ReplayBuffer

import time


# episode_ends: [ 22  54  76 114 137 184 205 257 292 331 381 429 499 521 581 624 692 720 751 795]

# [ 113  181  223  248  314  441  466  516  543  617  702  731  865 1019 1193 1228 1291 1330 1423 1499]

# def import_from_omni_after_app(): 


    



class IsaaclabRunner(BaseRunner):
    def __init__(self,
                output_dir,
                eval_episodes=20,
                max_steps=100,
                n_obs_steps=8,
                n_action_steps=8,
                fps=10,
                crf=22,
                render_size=84,
                tqdm_interval_sec=5.0,
                task_name=None,
                device="cuda:0",
                use_point_crop=True,
                use_contact_forces = True,
                num_points=512, 
                use_binary_contact = True,
                env_options: dict = None,
                ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.device = device
        self.use_point_crop = use_point_crop
        self.use_contact_forces = use_contact_forces
        self.num_points = num_points
        self.use_binary_contact = use_binary_contact

        # TODO: implement the "parser"
        parser = argparse.ArgumentParser(prog="isaaclab_runner.py", description="parser for IsaaclabRunner")
        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)
        # args_cli_tmp = parser.parse_args()  # type(args_cli_tmp): Namespace
        # parse_args(args, namespace), if args = none, then args = sys.args[1:], if namespace = none, then namespace = argparse.Namespace()
        # but we can only parse the added params through parser.parse_known_args(), so
        
        args_cli_tmp, exclusive_args = parser.parse_known_args()
        args_cli_dict = vars(args_cli_tmp)
        args_cli_dict.update(env_options)
        self.args_cli = argparse.Namespace(**args_cli_dict)

        # Launch the app
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app

        # import from omniverse, the imports must be done after the app is launched, or it will raise errors
        import omni.isaac.lab_tasks  # noqa: F401
        from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from omni.isaac.lab.utils.assets import retrieve_file_path
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from conditional_flow_matching.gym_util.isaaclab_point_cloud import PointcloudVisualizer
        
        # config environment and agents
        self.env_cfg = parse_env_cfg(
            self.args_cli.task, device=self.args_cli.device, num_envs=self.args_cli.num_envs, use_fabric=not self.args_cli.disable_fabric
        )
        self.agent_cfg = load_cfg_from_registry(self.args_cli.task, "rl_games_cfg_entry_point")
        
        # define env and agent
        rl_device = self.agent_cfg["params"]["config"]["device"]
        clip_obs = self.agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = self.agent_cfg["params"]["env"].get("clip_actions", math.inf)


        # create isaac environment
        assert(self.args_cli.task in [
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-v0",
     "Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-v0",
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-v0", 
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-SingleCam-v0",
     "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-v0",
     "ShadowInHandPushPCTactileSingleCam-v0",
     "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-Multi-Object-v0",
     "ShadowInHandPushDoublePCTactileSingleCam-v0",
     "ShadowInHandPushDoublePCTactileMultipleSuccess-v0",
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-ZAxis-v0",
     "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-GivenStep-v0",
     ])

        env = gym.make(self.args_cli.task, cfg=self.env_cfg, render_mode = "rgb_array")

        # wrap around environment for rl-games
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
                
        # register the env
        vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        env.seed(self.agent_cfg["params"]["seed"])

        # resume path is fixed now
        # 前三个env name
        if self.args_cli.task in [
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0", 
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0", 
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-SingleCam-v0",
            ]:
            resume_path = '/home/yijin/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/2024-09-20_16-02-25/nn/shadow_hand_openai_ff.pth'
        elif self.args_cli.task in [
            "Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-v0",
            "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-v0",
            "ShadowInHandPushPCTactileSingleCam-v0",
            "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-Multi-Object-v0",
            "ShadowInHandPushDoublePCTactileSingleCam-v0",
            "ShadowInHandPushDoublePCTactileMultipleSuccess-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-ZAxis-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-GivenStep-v0",
            ]:
            resume_path = '/home/yijin/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand/2024-09-21_21-40-28/nn/shadow_hand.pth'
        else:
            raise NotImplementedError(f"task: {self.args_cli.task} is not implemented yet")
        
        self.agent_cfg["params"]["load_checkpoint"] = True
        self.agent_cfg["params"]["load_path"] = resume_path

        # create runner from rl-games
        self.runner = Runner()
        self.runner.load(self.agent_cfg)
        self.agent = self.runner.create_player()

        
        # self.agent.restore(resume_path)
        self.agent.reset()

        self.camera_numbers = 2

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        self.rlenv = env

        self.env = MultiStepWrapper(
                SimpleVideoRecordingWrapper(env),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        

        # process the env
        self.env = self.preprocess_env(self.env)
        assert(isinstance(self.env.observation_space, gym.spaces.Dict))
        cprint(f"[IsaaclabRunner]: env.observation_space: {self.env.observation_space}", "green")
    
        # self.pc_generator = PointcloudGenerator()
    
    # def get_pc_and_color(self, obs, camera_numbers):

    #     point_cloud_list = []
    #     for obs_step in range(self.n_obs_steps):
    #         points_np, colors_np = self.pc_generator.get_pc_and_color(obs, obs_step, self.camera_numbers, self.num_points)

    #         point_cloud = torch.tensor(np.concatenate([points_np, colors_np], axis=1), dtype=torch.float32, device=self.device)
    #         point_cloud_list.append(point_cloud)
        
    #     pc_torch = torch.stack(point_cloud_list, dim=0)
    #     return pc_torch

    def preprocess_env(self, env):
        
        # defalt setting
        env.observation_space = gym.spaces.Dict({
            'agent_pos': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(24, ), 
                dtype=np.float32
            ),
            'point_cloud': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.num_points, 6), 
                dtype=np.float32
            
            ), 
            'goal_reached': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1, ), 
                dtype=np.bool8
            
            ),
            
        })

        if self.args_cli.task in [
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-SingleCam-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-ZAxis-v0",
            "Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-FixWrist-GivenStep-v0",
            ]:
            env.observation_space['obs'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(42, )
            )
            env.observation_space['states'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(187, ),
            )
                    
        elif self.args_cli.task in [
            "Isaac-Repose-Cube-Shadow-Direct-HandInit-PointCloud-Tactile-v0",
            "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-v0",
            "ShadowInHandPushPCTactileSingleCam-v0",
            "Isaac-Repose-Cube-Shadow-Direct-Face-Down-Reorient-PC-Tactile-Multi-Object-v0",
            "ShadowInHandPushDoublePCTactileSingleCam-v0",
            "ShadowInHandPushDoublePCTactileMultipleSuccess-v0",
            
            ]:
            env.observation_space['obs'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(187, )
            )
        else:
            raise NotImplementedError(f"[IsaaclabRunner]task: {self.args_cli.task} is not implemented yet")

        if self.use_contact_forces:
            env.observation_space['contact_forces'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(24, 3),
                dtype=np.float32
            )
        return env
    
    def stack_obs(self, obses:dict) -> dict:
        obses = dict_apply(obses, lambda x: x.squeeze(1))
        return obses
        


    def run(self, policy: BasePolicy, databuffer:ReplayBuffer =None, save_video=True):

        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        # reset environment
        # for k, v in obs_dict.items():
        '''
            key: agent_pos, value.shape: torch.Size([2, 24])
            key: goal_env_ids, value.shape: torch.Size([2, 0])
            key: obs, value.shape: torch.Size([2, 42])
            key: point_cloud, value.shape: torch.Size([2, 1024, 6])
        '''
        # obs_dict = env.reset()
        # if isinstance(obs_dict, dict):
        #     obs = obs_dict["obs"]
        # # required: enables the flag for batched observations
        # _ = self.agent.get_batch_size(obs, 1)

        # define pointcloud and fig debugger
        if self.args_cli.point_cloud_debug:
            # pointCloudVisualizer = PointcloudVisualizer()
            # pointCloudVisualizerInitialized = False
            o3d_pc = o3d.geometry.PointCloud()
            o3d_sampled = o3d.geometry.PointCloud()
        else:   
            o3d_pc = o3d.geometry.PointCloud()
            o3d_sampled = o3d.geometry.PointCloud()
        if self.args_cli.camera_debug:
            fig, axes = plt.subplots(self.camera_numbers, 2)
        
        # define lists for recording
        all_traj_rewards = []
        all_success_rates = []


        loss_list = []

         # for debug
        if databuffer is not None:
            state_all = torch.tensor(databuffer['state'], dtype=policy.dtype, device=policy.device)
            action_all = torch.tensor(databuffer['action'], dtype=policy.dtype, device=policy.device)
            agent_pos_all = torch.tensor(databuffer['agent_pos'], dtype=policy.dtype, device=policy.device)
            pc_all = torch.tensor(databuffer['point_cloud'], dtype=policy.dtype, device=policy.device)

        obs_last = None
        pc_last = None
        traj_begin_idx = 0
        traj_begin_idx_max = 113

        all_videos = []

        """Play with RL-Games agent."""
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in IsaacLab {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        for episode_idx in range(self.eval_episodes):
            assert(self.simulation_app.is_running())
            # start rollout
            # if episode_idx == 0:
            obs_dict = env.reset()
            policy.reset()
            obs = self.agent.obs_to_torch(obs_dict) 
            # required: enables the flag for batched observations
            _ = self.agent.get_batch_size(obs, 1)


            next_episode = False
            traj_reward = 0

            local_step = 0
            
            while not next_episode:
                local_step += 1
                obs_dict = dict(obs_dict)   # isaaclab is torch_based
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))

                obs_dict = self.stack_obs(obs_dict)  # (T, num_envs(1 for now), obs_shape) -> (T, obs_shape)
                # for k, v in obs_dict.items():

                obs = self.agent.obs_to_torch(obs_dict)[-1].squeeze(0).unsqueeze(0)  # get the last observation

                # for k, v in obs_dict.items():

                # for debug 
                if databuffer is not None:
                    traj_idx = min(traj_begin_idx_max, traj_begin_idx + local_step-1)
                    action_gt = action_all[traj_idx].unsqueeze(0).unsqueeze(0)
                    state_gt = state_all[traj_idx].unsqueeze(0)
                    agent_pos_gt = agent_pos_all[traj_idx].unsqueeze(0)
                    pc_gt = pc_all[traj_idx].unsqueeze(0)

                # st = time.time()
                with torch.inference_mode(): 
                    obs_dict_input = {}

                    # modified for debug
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)

                    if policy.use_contact_forces:
                        obs_dict_input['contact_forces'] = obs_dict['contact_forces'].unsqueeze(0)
                        # if self.use_binary_contact:
                        #     obs_dict_input['contact_forces'] = torch.norm(obs_dict_input['contact_forces'], dim=-1, keepdim=True).squeeze(-1)
                        #     obs_dict_input['contact_forces'] = (obs_dict_input['contact_forces'] > 0.1).float()

                    pc_debug = obs_dict['point_cloud'][0]
                    # import open3d as o3d
                    # o3d_pc = o3d.geometry.PointCloud()
                    # o3d_pc.points = o3d.utility.Vector3dVector(pc_debug[..., :3].cpu().numpy())
                    # o3d_pc.colors = o3d.utility.Vector3dVector(pc_debug[..., 3:].cpu().numpy() / 255.0)
                    # o3d.visualization.draw_geometries([o3d_pc])

                    # if local_step == 1: 
                    # obs_dict_input['point_cloud'] = pc_gt.repeat(self.n_obs_steps, 1, 1).unsqueeze(0)
                    # obs_dict_input['agent_pos'] = agent_pos_gt.repeat(self.n_obs_steps, 1).unsqueeze(0)

                    # obs_dict_input['point_cloud'] = pc_gt
                    # obs_dict_input['agent_pos'] = agent_pos_gt
                    # for k, v in obs_dict_input.items():
                    '''
                    key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
                    key: agent_pos, value.shape: torch.Size([1, 2, 24])
                    '''
                    # ac_st  = time.time()
                    action_dict = policy.predict_action(obs_dict_input)
                    # ac_et = time.time()
                    
                    
                
                if local_step == 1 and databuffer is not None: 
                    action_rl = self.agent.get_action(obs, is_deterministic=True)
                    action_rl = action_rl.unsqueeze(0)   
                    states_check = obs_dict['states'][-1]
                    obs_check = obs_dict['obs'][-1]
                    pc_current = obs_dict['point_cloud'][-1].squeeze(0).detach().cpu()
                    obs_current = obs_dict['obs'][-1].squeeze(0).detach().cpu()
                    agent_pos_check = obs_dict['agent_pos'][-1].squeeze(0).detach().cpu()
                    cprint(f"traj_idx: {traj_idx}", "red")
                    obj_pos = states_check[..., 48:51].squeeze(0).detach().cpu()
                    obj_rot = states_check[..., 51:55].squeeze(0).detach().cpu()
                    obj_pos_pred = obs_check[..., 15:18].squeeze(0).detach().cpu()

                    obj_gt = state_gt[..., 48:51].squeeze(0).detach().cpu()
                    obj_rot_gt = state_gt[..., 51:55].squeeze(0).detach().cpu()
                    cprint(f"[IsaaclabRunnercfm]: obj_gt: {obj_gt}, obj_rot_gt: {obj_rot_gt}", "magenta")
                    cprint(f"[IsaaclabRunnercfm]: obj_pos: {obj_pos}, obj_rot: {obj_rot}", "green")
                    # assert(torch.all(obj_pos == obj_pos_pred))
                    # assert(torch.all(torch.isclose(obj_pos, obj_gt, atol=1e-4)))
                    
                    if pc_last != None: 
                        # compute the iou bettwen the two point clouds
                        pc_last = pc_last.squeeze(0).detach().cpu()
                        
                        # visualize the pc
                        o3d_pc.points = o3d.utility.Vector3dVector(pc_last[..., :3].numpy())
                        o3d_pc.colors = o3d.utility.Vector3dVector(pc_last[..., 3:].numpy())
                        o3d_sampled.points = o3d.utility.Vector3dVector(pc_current[..., :3].numpy())
                        o3d_sampled.colors = o3d.utility.Vector3dVector(pc_current[..., 3:].numpy())
                        # o3d.visualization.draw_geometries([o3d_pc])
                        # o3d.visualization.draw_geometries([o3d_sampled])
                        # compute chamfer distance
                    import chamferdist
                    chamferDist = chamferdist.ChamferDistance()
                    cprint(f"pc_gt.shape: {pc_gt.shape}, pc_current.shape: {pc_current.shape}", "magenta")
                    chamfer_distance = chamferDist(pc_gt[..., :3].detach().cpu().float(), pc_current[..., :3].unsqueeze(0).float())
                    cprint(f"[IsaaclabRunnercfm]: chamfer_distance: {chamfer_distance}", "magenta")

                    pc_loss = torch.nn.functional.mse_loss(pc_gt.detach().cpu().float(), pc_current.unsqueeze(0).float())

                    # obs_loss = torch.nn.functional.mse_loss(obs_last, obs_current)

                # if local_step == 1 or local_step == 15:    
                    obs_last = obs_current
                    pc_last = pc_current
                        

                    # visualize the pc
                    point_cloud_check = obs_dict['point_cloud'][-1].squeeze(0).detach().cpu()
                    o3d_pc_check = o3d.geometry.PointCloud()
                    o3d_pc_check.points = o3d.utility.Vector3dVector(point_cloud_check[..., :3].numpy())
                    o3d_pc_check.colors = o3d.utility.Vector3dVector(point_cloud_check[..., 3:].numpy() / 255.0 )
                    o3d.visualization.draw_geometries([o3d_pc_check])
                
                torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
                # action = torch_action_dict['action'].squeeze(0)
                action = torch_action_dict['action'].squeeze(0).unsqueeze(1)

                if databuffer is not None:
                    loss_actions = torch.nn.functional.mse_loss(action.squeeze(0).squeeze(0).detach().cpu(), action_rl.squeeze(0).squeeze(0).detach().cpu())
                    loss_actions_2 = torch.nn.functional.mse_loss(action_gt.squeeze(0).squeeze(0).detach().cpu(), action_rl.squeeze(0).squeeze(0).detach().cpu())
                    lossaction_3 = torch.nn.functional.mse_loss(action_gt.squeeze(0).squeeze(0).detach().cpu(), action.squeeze(0).squeeze(0).detach().cpu())
                    if local_step < traj_begin_idx_max - traj_begin_idx:   # 这一段轨迹只到22, 大于的话没有意义
                        pass
                    loss_list.append(loss_actions.item())



                obs_dict, reward, done, extras = env.step(action)
                # obs_dict, reward, done, extras = env.step(action_rl)
                # obs_dict, reward, done, extras = env.step(action_gt)

                obs = self.agent.obs_to_torch(obs_dict) 
                # obs.to(device=device)
                traj_reward += reward.item()
                success_obs_dict = env._get_obs(max(3, self.n_obs_steps))  # 3 is because the decimation of cube is 3

                success = torch.any(success_obs_dict['goal_reached']).item()
                if success:
                    next_episode = True

                    if self.args_cli.task in [
                        "ShadowInHandPushDoublePCTactileMultipleSuccess-v0",
                        ]:
                        next_episode = False
                    all_success_rates.append(success)
                    cprint(f"Success in episode {episode_idx}, reward: {reward}, local_step: {local_step}", "green")
                if done : # timeout or failed
                    next_episode = True
                    all_success_rates.append(success)  # failed
                    cprint(f"Episode: {episode_idx} failed, reward: {reward}, local_step: {local_step}", "red")

                    # reset env manually, if success, we don't need to because the reset_idx() is now written inside the env
                    # env.reset()
                
                
                all_traj_rewards.append(traj_reward)

                # ed = time.time()

            if episode_idx % 3 == 0:  # record vedio every 2 episodes
            
                videos = env.env.get_video()
                cprint(f"record vedio in episode: {episode_idx}, video.shape: {videos.shape}", "cyan") #(T, C, H, W)
                if len(videos.shape) == 5:
                    videos = videos[:, 0]  # select first frame
                
                if save_video:
                    all_videos.append(videos)
                    # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
                    
                
                # clear video
                videos = None
        
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        cprint(f"all_success_rates: {all_success_rates}", 'green')
        cprint(f"max_steps: {self.max_steps}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()


        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            all_videos.append(videos)
            all_videos = np.concatenate(all_videos, axis=0)   # concatinate all "T" frames
            cprint(f"type(all_videos): {type(all_videos)}, all_videos.shape: {all_videos.shape}", "green")
            # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            videos_wandb = wandb.Video(all_videos, fps=self.fps, format="mp4")

            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        loss_arr = np.array(loss_list)
        # draw the distribution of loss
        # plt.hist(loss_arr, bins=100, color='blue', alpha=0.7)
        # plt.title("loss distribution")

        
        # plt.show()

        return log_data
    


    def run_gt_action(self, policy: BasePolicy, databuffer:ReplayBuffer =None, save_video=True):

        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        # reset environment
        # for k, v in obs_dict.items():
        '''
            key: agent_pos, value.shape: torch.Size([2, 24])
            key: goal_env_ids, value.shape: torch.Size([2, 0])
            key: obs, value.shape: torch.Size([2, 42])
            key: point_cloud, value.shape: torch.Size([2, 1024, 6])
        '''
        # obs_dict = env.reset()
        # if isinstance(obs_dict, dict):
        #     obs = obs_dict["obs"]
        # # required: enables the flag for batched observations
        # _ = self.agent.get_batch_size(obs, 1)

        # define pointcloud and fig debugger
        if self.args_cli.point_cloud_debug:
            pointCloudVisualizer = PointcloudVisualizer()
            pointCloudVisualizerInitialized = False
            o3d_pc = o3d.geometry.PointCloud()
            o3d_sampled = o3d.geometry.PointCloud()
        else:   
            o3d_pc = o3d.geometry.PointCloud()
            o3d_sampled = o3d.geometry.PointCloud()
        if self.args_cli.camera_debug:
            fig, axes = plt.subplots(self.camera_numbers, 2)
        
        # define lists for recording
        all_traj_rewards = []
        all_success_rates = []


        loss_list = []

         # for debug
        if databuffer is not None:
            state_all = torch.tensor(databuffer['state'], dtype=policy.dtype, device=policy.device)
            action_all = torch.tensor(databuffer['action'], dtype=policy.dtype, device=policy.device)
            agent_pos_all = torch.tensor(databuffer['agent_pos'], dtype=policy.dtype, device=policy.device)
            pc_all = torch.tensor(databuffer['point_cloud'], dtype=policy.dtype, device=policy.device)

        obs_last = None
        pc_last = None
        traj_begin_idx = 0
        traj_begin_idx_max = 113


        """Play with RL-Games agent."""
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in IsaacLab {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        for episode_idx in range(self.eval_episodes):
            assert(self.simulation_app.is_running())
            # start rollout
            obs_dict = env.reset()
            policy.reset()
            obs = self.agent.obs_to_torch(obs_dict) 
            # required: enables the flag for batched observations
            _ = self.agent.get_batch_size(obs, 1)


            next_episode = False
            traj_reward = 0

            local_step = 0
            
            while not next_episode:
                local_step += 1
                obs_dict = dict(obs_dict)   # isaaclab is torch_based
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))

                obs_dict = self.stack_obs(obs_dict)  # (T, num_envs(1 for now), obs_shape) -> (T, obs_shape)
                # for k, v in obs_dict.items():

                obs = self.agent.obs_to_torch(obs_dict)[-1].squeeze(0).unsqueeze(0)  # get the last observation

                # for k, v in obs_dict.items():

                # for debug 
                
                traj_idx = min(traj_begin_idx_max, traj_begin_idx + local_step-1)
                action_gt = action_all[traj_idx].unsqueeze(0).unsqueeze(0)
                state_gt = state_all[traj_idx].unsqueeze(0)
                agent_pos_gt = agent_pos_all[traj_idx].unsqueeze(0)
                pc_gt = pc_all[traj_idx].unsqueeze(0)

                with torch.inference_mode(): 
                    obs_dict_input = {}

                    # modified for debug
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)

                    # if local_step == 1: 
                    # obs_dict_input['point_cloud'] = pc_gt.repeat(self.n_obs_steps, 1, 1).unsqueeze(0)
                    # obs_dict_input['agent_pos'] = agent_pos_gt.repeat(self.n_obs_steps, 1).unsqueeze(0)

                    # obs_dict_input['point_cloud'] = pc_gt
                    # obs_dict_input['agent_pos'] = agent_pos_gt
                    # for k, v in obs_dict_input.items():
                    '''
                    key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
                    key: agent_pos, value.shape: torch.Size([1, 2, 24])
                    '''
                    action_dict = policy.predict_action(obs_dict_input)
                    action_rl = self.agent.get_action(obs, is_deterministic=True)
                    action_rl = action_rl.unsqueeze(0)
                    
                    
                
                states_check = obs_dict['states'][-1]
                obs_check = obs_dict['obs'][-1]
                pc_current = obs_dict['point_cloud'][-1].squeeze(0).detach().cpu()
                obs_current = obs_dict['obs'][-1].squeeze(0).detach().cpu()
                agent_pos_check = obs_dict['agent_pos'][-1].squeeze(0).detach().cpu()
                if local_step == 1 : 
                    cprint(f"traj_idx: {traj_idx}", "red")
                    obj_pos = states_check[..., 48:51].squeeze(0).detach().cpu()
                    obj_rot = states_check[..., 51:55].squeeze(0).detach().cpu()
                    obj_pos_pred = obs_check[..., 15:18].squeeze(0).detach().cpu()

                    obj_gt = state_gt[..., 48:51].squeeze(0).detach().cpu()
                    obj_rot_gt = state_gt[..., 51:55].squeeze(0).detach().cpu()
                    cprint(f"[IsaaclabRunnercfm]: obj_gt: {obj_gt}, obj_rot_gt: {obj_rot_gt}", "magenta")
                    cprint(f"[IsaaclabRunnercfm]: isclose: {torch.isclose(obj_pos, obj_gt, atol=1e-4)}, {torch.isclose(obj_rot, obj_rot_gt, atol=1e-4)}", "magenta")
                    cprint(f"[IsaaclabRunnercfm]: obj_pos: {obj_pos}, obj_rot: {obj_rot}", "green")

                    cprint(f"[IsaaclabRunnercfm]: agent_pos_check: {agent_pos_check}", "magenta")
                    cprint(f"[IsaaclabRunnercfm]: agent_pos_gt: {agent_pos_gt}", "magenta")
                    assert(torch.all(obj_pos == obj_pos_pred))
                    assert(torch.all(torch.isclose(obj_pos, obj_gt, atol=1e-4)))
                    
                    if pc_last != None: 
                        # compute the iou bettwen the two point clouds
                        pc_last = pc_last.squeeze(0).detach().cpu()
                        
                        # visualize the pc
                        o3d_pc.points = o3d.utility.Vector3dVector(pc_last[..., :3].numpy())
                        o3d_pc.colors = o3d.utility.Vector3dVector(pc_last[..., 3:].numpy())
                        o3d_sampled.points = o3d.utility.Vector3dVector(pc_current[..., :3].numpy())
                        o3d_sampled.colors = o3d.utility.Vector3dVector(pc_current[..., 3:].numpy())
                        # o3d.visualization.draw_geometries([o3d_pc])
                        # o3d.visualization.draw_geometries([o3d_sampled])
                        # compute chamfer distance
                    import chamferdist
                    chamferDist = chamferdist.ChamferDistance()
                    cprint(f"pc_gt.shape: {pc_gt.shape}, pc_current.shape: {pc_current.shape}", "magenta")
                    chamfer_distance = chamferDist(pc_gt[..., :3].detach().cpu().float(), pc_current[..., :3].unsqueeze(0).float())
                    cprint(f"[IsaaclabRunnercfm]: chamfer_distance: {chamfer_distance}", "magenta")

                    cprint(f"range of pc_current: {torch.min(pc_current[..., :3])}, {torch.max(pc_current[..., :3])}", "blue")
                    cprint(f"range_color of pc_current: {torch.min(pc_current[..., 3:])}, {torch.max(pc_current[..., 3:])}", "blue")
                    cprint(f"range_color of pc_gt: {torch.min(pc_gt[..., 3:])}, {torch.max(pc_gt[..., 3:])}", "blue")
                    cprint(f"range of pc_gt: {torch.min(pc_gt[..., :3])}, {torch.max(pc_gt[..., :3])}", "blue")

                    pc_loss = torch.nn.functional.mse_loss(pc_gt.detach().cpu().float(), pc_current.unsqueeze(0).float())
                    cprint(f"[IsaaclabRunnercfm]: pc_loss: {pc_loss}", "magenta")

                    # obs_loss = torch.nn.functional.mse_loss(obs_last, obs_current)

                # if local_step == 1 or local_step == 15:    
                    obs_last = obs_current
                    pc_last = pc_current
                        

                    # visualize the pc
                    # point_cloud_check = obs_dict['point_cloud'][-1].squeeze(0).detach().cpu()
                    # o3d_pc_check = o3d.geometry.PointCloud()
                    # o3d_pc_check.points = o3d.utility.Vector3dVector(point_cloud_check[..., :3].numpy())
                    # o3d_pc_check.colors = o3d.utility.Vector3dVector(point_cloud_check[..., 3:].numpy() )
                    # o3d.visualization.draw_geometries([o3d_pc_check])
                
                torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
                # action = torch_action_dict['action'].squeeze(0)
                action = torch_action_dict['action'].squeeze(0).unsqueeze(1)

                loss_actions = torch.nn.functional.mse_loss(action.squeeze(0).squeeze(0).detach().cpu(), action_rl.squeeze(0).squeeze(0).detach().cpu())
                loss_actions_2 = torch.nn.functional.mse_loss(action_gt.squeeze(0).squeeze(0).detach().cpu(), action_rl.squeeze(0).squeeze(0).detach().cpu())
                lossaction_3 = torch.nn.functional.mse_loss(action_gt.squeeze(0).squeeze(0).detach().cpu(), action.squeeze(0).squeeze(0).detach().cpu())
                if local_step < traj_begin_idx_max - traj_begin_idx:   # 这一段轨迹只到22, 大于的话没有意义
                    cprint(f"local_step: {local_step}, traj_idx: {traj_idx} loss_actions:{loss_actions_2.item()}", "yellow", end=" \n" )
                loss_list.append(loss_actions.item())



                # obs_dict, reward, done, extras = env.step(action)
                # obs_dict, reward, done, extras = env.step(action_rl)
                obs_dict, reward, done, extras = env.step(action_gt)

                obs = self.agent.obs_to_torch(obs_dict) 
                # obs.to(device=device)
                traj_reward += reward.item()

                success = torch.any(obs_dict['goal_reached']).item()
                if success:
                    next_episode = True
                    all_success_rates.append(success)
                    cprint(f"Success in episode {episode_idx}, reward: {reward}, local_step: {local_step}", "green")
                if done : # timeout or failed
                    next_episode = True
                    all_success_rates.append(success)  # failed
                    cprint(f"Episode: {episode_idx} failed, reward: {reward}, local_step: {local_step}", "red")
                
                
                all_traj_rewards.append(traj_reward)
        
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        cprint(f"all_success_rates: {all_success_rates}", 'green')
        cprint(f"max_steps: {self.max_steps}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()


        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        loss_arr = np.array(loss_list)
        # draw the distribution of loss
        # plt.hist(loss_arr, bins=100, color='blue', alpha=0.7)
        # plt.title("loss distribution")
        cprint(f"avg_loss_arr: {np.mean(loss_arr)}, std_loss_arr: {np.std(loss_arr)}", "green")
        cprint(f"loss_arr range: {np.min(loss_arr)} ~ {np.max(loss_arr)}", "green")

        
        # plt.show()

        return log_data
    


    from conditional_flow_matching.common.replay_buffer import ReplayBuffer
    def run_training(self, policy: BasePolicy, databuffer:ReplayBuffer, save_video=True):

        device = policy.device
        dtype = policy.dtype

        obs = torch.tensor(databuffer['obs'], dtype=policy.dtype, device=policy.device)
        # states = torch.tensor(databuffer['state'], dtype=policy.dtype, device=policy.device)
        gt_actions = torch.tensor(databuffer['action'], dtype=policy.dtype, device=policy.device)
        point_clouds = torch.tensor(databuffer['point_cloud'], dtype=policy.dtype, device=policy.device)
        agent_pos = torch.tensor(databuffer['agent_pos'], dtype=policy.dtype, device=policy.device)

        assert(obs.shape[0] == gt_actions.shape[0] and obs.shape[0] == point_clouds.shape[0] and obs.shape[0] == agent_pos.shape[0])

        '''
        loss_1:0.0 loss_2:0.0018687315750867128 loss_3:0.0018687315750867128
        loss_1:0.0 loss_2:0.001234753872267902 loss_3:0.001234753872267902   # 如果种子确实能固定，那么rl_model预测的应该就是gt
        loss_1:0.0 loss_2:0.0025277198292315006 loss_3:0.0025277198292315006
        '''


        # num_gts = states.shape[0]

        num_gts = obs.shape[0]

        loss_list_1 = []
        loss_list_2 = []
        loss_list_3 = []

        for i in range(1, num_gts):
            
            # action prediction for rl models
            obs_rl = obs[i].unsqueeze(0)
            _ = self.agent.get_batch_size(obs_rl, 1)
            action_rl = self.agent.get_action(obs_rl, is_deterministic=True)
            
            # TODO modify to find the loss between these two
            with torch.inference_mode():
                obs_dict_input = {}
                obs_dict_input["point_cloud"] = point_clouds[i-self.n_obs_steps+1: i+1, ...].unsqueeze(0)  # 
                # 用unsqueeze是为了给他加上一个"batch size" 因为训练的时候有batch veryfy 的时候也应该有
                obs_dict_input["agent_pos"] = agent_pos[i-self.n_obs_steps+1: i+1, ...].unsqueeze(0)
                action_dict = policy.predict_action(obs_dict_input) # action for next step

            action_dp = action_dict['action'].squeeze(0)

            # obs1 记录的 action 就是 time 2 的action
            loss_1 = torch.nn.functional.mse_loss(action_rl.detach().cpu(), gt_actions[i].detach().cpu())
            loss_2 = torch.nn.functional.mse_loss(action_dp.detach().cpu(), gt_actions[i].detach().cpu())

            loss_3 = torch.nn.functional.mse_loss(action_rl.detach().cpu(), action_dp.detach().cpu())

            cprint(f"loss_1:{loss_1.item()}", "yellow", end=" " )
            cprint(f"loss_2:{loss_2.item()}", "cyan", end=" " )
            cprint(f"loss_3:{loss_3.item()}", "magenta", end=" " )

            loss_list_1.append(loss_1.item())
            loss_list_2.append(loss_2.item())
            loss_list_3.append(loss_3.item())


        

        # conver loss lists to numpy
        loss_arr_1 = np.array(loss_list_1)
        loss_arr_2 = np.array(loss_list_2)
        loss_arr_3 = np.array(loss_list_3)

        


        # draw the loss_2's distribution using histogram
        plt.hist(loss_arr_2, bins=100, color='blue', alpha=0.7)

        cprint(f"###### conclusion #######", "green")

        cprint(f"avg_loss_arr_1: {np.mean(loss_arr_1)}, std_loss_arr_1: {np.std(loss_arr_1)}", "green")
        cprint(f"loss_arr_1 range: {np.min(loss_arr_1)} ~ {np.max(loss_arr_1)}", "green")

        cprint(f"avg_loss_arr_2: {np.mean(loss_arr_2)}, std_loss_arr_2: {np.std(loss_arr_2)}", "green")
        cprint(f"loss_arr_2 range: {np.min(loss_arr_2)} ~ {np.max(loss_arr_2)}", "green")

        cprint(f"avg_loss_arr_3: {np.mean(loss_arr_3)}, std_loss_arr_3: {np.std(loss_arr_3)}", "green")
        cprint(f"loss_arr_3 range: {np.min(loss_arr_3)} ~ {np.max(loss_arr_3)}", "green")

        # add title to the histogram
        plt.title("loss_2's distribution")
        plt.show()
    

    def run_begin(self, policy: BasePolicy, databuffer:ReplayBuffer, save_video=True):
        device = policy.device
        dtype = policy.dtype

        obs = torch.tensor(databuffer['obs'], dtype=policy.dtype, device=policy.device)
        # states = torch.tensor(databuffer['state'], dtype=policy.dtype, device=policy.device)
        gt_actions = torch.tensor(databuffer['action'], dtype=policy.dtype, device=policy.device)
        point_clouds = torch.tensor(databuffer['point_cloud'], dtype=policy.dtype, device=policy.device)
        agent_pos = torch.tensor(databuffer['agent_pos'], dtype=policy.dtype, device=policy.device)

        episode_ends = torch.tensor(databuffer.root['meta']['episode_ends'], dtype=policy.dtype, device=policy.device)

        assert(obs.shape[0] == gt_actions.shape[0] and obs.shape[0] == point_clouds.shape[0] and obs.shape[0] == agent_pos.shape[0])

        for i in episode_ends[...,:-1]:
            i = int(i)
            cprint(f"episode_ends: {i}", "green")
            

            obs_rl = obs[i].unsqueeze(0)
            _ = self.agent.get_batch_size(obs_rl, 1)
            action_rl = self.agent.get_action(obs_rl, is_deterministic=True)

            with torch.inference_mode():
                obs_dict_input = {}
                # obs_dict_input["point_cloud"] = point_clouds[i-self.n_obs_steps+1: i+1, ...].unsqueeze(0)  # 
                # 用unsqueeze是为了给他加上一个"batch size" 因为训练的时候有batch veryfy 的时候也应该有
                # obs_dict_input["agent_pos"] = agent_pos[i-self.n_obs_steps+1: i+1, ...].unsqueeze(0)
                obs_dict_input["point_cloud"] = point_clouds[i, ...].repeat(self.n_obs_steps, 1, 1).unsqueeze(0)
                obs_dict_input["agent_pos"] = agent_pos[i, ...].repeat(self.n_obs_steps, 1).unsqueeze(0)
                cprint(f"agent_pos: {obs_dict_input['agent_pos']}", "green")
                action_dict = policy.predict_action(obs_dict_input) # action for next step
            
            action_dp = action_dict['action'].squeeze(0)

            loss_1 = torch.nn.functional.mse_loss(action_rl.squeeze(0).detach().cpu(), gt_actions[i].detach().cpu())
            loss_2 = torch.nn.functional.mse_loss(action_dp.squeeze(0).detach().cpu(), gt_actions[i].detach().cpu())

            cprint(f"loss_1:{loss_1.item()}", "yellow", end=" " )
            cprint(f"loss_2:{loss_2.item()}", "cyan", end=" \n " )



        

        # """Play with RL-Games agent."""
        # # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        # for episode_idx in range(self.eval_episodes):
        #     assert(self.simulation_app.is_running())
        #     # start rollout
        #     obs_dict = env.reset()
        #     policy.reset()
        #     obs = self.agent.obs_to_torch(obs_dict) 

        #     # required: enables the flag for batched observations
        #     _ = self.agent.get_batch_size(obs, 1)


        #     next_episode = False
        #     traj_reward = 0

        #     local_step = 0
            

        #     while not next_episode:
        #         local_step += 1
        #         # cprint(f"local_step: {local_step}", "green")
        #         obs_dict = dict(obs_dict)   # isaaclab is torch_based
        #         obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                
        #         obs_dict = self.stack_obs(obs_dict)  # (T, num_envs(1 for now), obs_shape) -> (T, obs_shape)

        #         obs = self.agent.obs_to_torch(obs_dict)[-1].squeeze(0).unsqueeze(0)  # get the last observation
        #         # cprint(f"obs: {obs.shape}, {obs.device}", "green")

        #         # for k, v in obs_dict.items():
        #         #     cprint(f"key: {k}, value.shape: {v.shape}", "green")

        #         with torch.inference_mode():
        #             obs_dict_input = {}
        #             obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
        #             obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
        #             # for k, v in obs_dict_input.items():
        #             #     cprint(f"key: {k}, value.shape: {v.shape}", "yellow")   
        #             '''
        #             key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
        #             key: agent_pos, value.shape: torch.Size([1, 2, 24])
        #             '''
        #             action_dict = policy.predict_action(obs_dict_input)
                
        #         torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
        #         action = torch_action_dict['action'].squeeze(0)


        #         action_rl = self.agent.get_action(obs, is_deterministic=True)

        #         loss_actions = torch.nn.functional.mse_loss(action.detach().cpu(), action_rl.detach().cpu())
        #         loss_list.append(loss_actions.item())



        #         # obs_dict, reward, done, extras = env.step(action)
        #         obs_dict, reward, done, extras = env.step(action_rl)

        #         obs = self.agent.obs_to_torch(obs_dict) 
        #         # obs.to(device=device)
        #         # cprint(f"obs: {obs.shape}, {obs.device}", "green")
        #         # cprint(f"done: {done}", "yellow")
        #         traj_reward += reward.item()

                
        #         success = torch.any(obs_dict['goal_reached']).item()
        #         if success:
        #             next_episode = True
        #             all_success_rates.append(success)
                
        #         # cprint(f"done: {done}", "yellow")
                
        #         if done : # timeout or failed
        #             next_episode = True
        #             all_success_rates.append(success)  # failed
                
                
        #         all_traj_rewards.append(traj_reward)
        
        # log_data = dict()

        # log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        # log_data['mean_success_rates'] = np.mean(all_success_rates)

        # log_data['test_mean_score'] = np.mean(all_success_rates)

        # self.logger_util_test.record(np.mean(all_success_rates))
        # self.logger_util_test10.record(np.mean(all_success_rates))
        # log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        # log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()


        # videos = env.env.get_video()
        # if len(videos.shape) == 5:
        #     videos = videos[:, 0]  # select first frame
        
        # if save_video:
        #     videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        #     log_data[f'sim_video_eval'] = videos_wandb

        # _ = env.reset()
        # videos = None

        # loss_arr = np.array(loss_list)

        # return log_data
    



        # def run_with_start_pose(self, policy: BasePolicy, start_pos: torch.Tensor, start_rot: torch.Tensor,  save_video=True):

    #     self.env = MultiStepWrapper(
    #             env,
    #             n_obs_steps=self.n_obs_steps,
    #             n_action_steps=self.n_action_steps,
    #             max_episode_steps=self.max_steps,
    #             reward_agg_method='sum',
    #         )

    #     # process the env
    #     self.env = self.preprocess_env(self.rlenv)
        
    #     assert(isinstance(self.env.observation_space, gym.spaces.Dict))
        
    #     assert(start_pos.shape[0] == start_rot.shape[0])

    #     eval_episodes = start_pos.shape[0]

    #     device = policy.device
    #     dtype = policy.dtype

    #     all_traj_rewards = []
    #     all_success_rates = []
    #     env = self.env

    #     # reset environment
    #     obs_dict = env.reset_with_pose(start_pos[0], start_rot[0])
    #     # cprint(f"obs_dict.keys(): {obs_dict.keys()}", "red")
    #     # for k, v in obs_dict.items():
    #     #     cprint(f"key: {k}, value.shape: {v.shape}", "green")
    #     '''
    #         key: agent_pos, value.shape: torch.Size([2, 24])
    #         key: goal_env_ids, value.shape: torch.Size([2, 0])
    #         key: obs, value.shape: torch.Size([2, 42])
    #         key: point_cloud, value.shape: torch.Size([2, 1024, 6])
    #     '''
    #     if isinstance(obs_dict, dict):
    #         obs = obs_dict["obs"]
    #     # required: enables the flag for batched observations
    #     _ = self.agent.get_batch_size(obs, 1)
        
    #     # define lists for recording
    #     all_traj_rewards = []
    #     all_success_rates = []


    #     """Play with RL-Games agent."""
    #     for episode_idx in tqdm.tqdm(range(eval_episodes), desc=f"Eval in IsaacLab {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
    #         assert(self.simulation_app.is_running())
    #         # start rollout
    #         obs_dict = env.reset_with_pose(start_pos[episode_idx], start_rot[episode_idx])
    #         states = obs_dict['states']
    #         obj_pos = states[..., 48:51]
    #         obj_rot = states[..., 51:55]
    #         policy.reset()
    #         obs = self.agent.obs_to_torch(obs_dict) 


    #         next_episode = False
    #         traj_reward = 0

    #         local_step = 0
            

    #         while not next_episode:
    #             pos_rot_idx = (episode_idx + 1) if episode_idx + 1 != episode_idx else episode_idx
    #             local_step += 1
    #             obs_dict = dict(obs_dict)   # isaaclab is torch_based
    #             obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))

    #             obs_dict = self.stack_obs(obs_dict)  # (T, num_envs(1 for now), obs_shape) -> (T, obs_shape)

    #             # for k, v in obs_dict.items():
    #             #     cprint(f"key: {k}, value.shape: {v.shape}", "green")

    #             with torch.inference_mode():
    #                 obs_dict_input = {}
    #                 obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
    #                 obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
    #                 # for k, v in obs_dict_input.items():
    #                 #     cprint(f"key: {k}, value.shape: {v.shape}", "yellow")   
    #                 '''
    #                 key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
    #                 key: agent_pos, value.shape: torch.Size([1, 2, 24])
    #                 '''
    #                 action_dict = policy.predict_action(obs_dict_input)
                
    #             torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
    #             action = torch_action_dict['action'].squeeze(0)

    #             obs_dict, reward, done, extras = env.step_with_pose(action, start_pos[pos_rot_idx], start_rot[pos_rot_idx])
    #             # cprint(f"done: {done}", "yellow")
    #             traj_reward += reward.item()

                
    #             success = torch.any(obs_dict['goal_reached']).item()
    #             if success:
    #                 next_episode = True
    #                 all_success_rates.append(success)
                
    #             # cprint(f"done: {done}", "yellow")
    #             if done : # timeout or failed
    #                 next_episode = True
    #                 all_success_rates.append(success)  # failed
                
                
    #             all_traj_rewards.append(traj_reward)
        
    #     log_data = dict()

    #     log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
    #     log_data['mean_success_rates'] = np.mean(all_success_rates)

    #     log_data['test_mean_score'] = np.mean(all_success_rates)

    #     self.logger_util_test.record(np.mean(all_success_rates))
    #     self.logger_util_test10.record(np.mean(all_success_rates))
    #     log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
    #     log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()


    #     videos = env.env.get_video()
    #     if len(videos.shape) == 5:
    #         videos = videos[:, 0]  # select first frame
        
    #     if save_video:
    #         videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
    #         log_data[f'sim_video_eval'] = videos_wandb

    #     _ = env.reset_with_pose(start_pos[episode_idx], start_rot[episode_idx])
    #     videos = None

    #     return log_data
    

    def close_simulation_app(self): 
        self.simulation_app.close()



