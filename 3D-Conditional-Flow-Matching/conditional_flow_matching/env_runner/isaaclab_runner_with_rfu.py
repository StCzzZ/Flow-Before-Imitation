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

import conditional_flow_matching.common.logger_util as logger_util
from conditional_flow_matching.gym_util.rfu_multistep_wrapper import RFUMultiStepWrapper
import time





class IsaaclabRunnerWithRFU(BaseRunner):
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
        self.num_points = num_points
        self.use_binary_contact = use_binary_contact

        # TODO: implement the "parser"
        parser = argparse.ArgumentParser(prog="isaaclab_runner.py", description="parser for IsaaclabRunner")
        # append AppLauncher cli args
        AppLauncher.add_app_launcher_args(parser)

        
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

        self.camera_numbers = 1

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

        self.env_rfu = RFUMultiStepWrapper(
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
            reward_agg_method='sum',
            object_name=task_name,
            device=self.device,
            vec_dim=1,
        )


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

        '''
            key: agent_pos, value.shape: torch.Size([2, 24])
            key: goal_env_ids, value.shape: torch.Size([2, 0])
            key: obs, value.shape: torch.Size([2, 42])
            key: point_cloud, value.shape: torch.Size([2, 1024, 6])
        '''
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
        all_videos = []

        """Play with RL-Games agent."""
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in IsaacLab {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        for episode_idx in range(self.eval_episodes):
            assert(self.simulation_app.is_running())
            # start rollout
            # if episode_idx == 0:
            obs_dict = env.reset()
            done = None
            policy.reset()
            next_episode = False
            traj_reward = 0
            local_step = 0
            
            while not next_episode:
                local_step += 1
                obs_dict = dict(obs_dict)   # isaaclab is torch_based
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                obs_dict = self.stack_obs(obs_dict)  # (T, num_envs(1 for now), obs_shape) -> (T, obs_shape)

                # st = time.time()
                with torch.inference_mode(): 
                    obs_dict_input = {}
                    # modified for debug
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    pc_debug = obs_dict['point_cloud'].squeeze(0)[-1].cpu().numpy()

                    # 反正这一段的宗旨就是,你obs里面有啥我就给你放一遍啥,考虑到现在其实action step  > n_obs_step, 所以其实是不亏的
                    if "OpenAI" in self.args_cli.task and "Direct" in self.args_cli.task:  # use openai env, we need to check the noise, i.e., obs[state]'s obj_pos == obs[obs]'s obj_pos
                        state_check = obs_dict['states'].squeeze(0)
                        joint_state = obs_dict['agent_pos'].squeeze(0)
                        obj_pos = state_check[..., 48:51].view(-1, 3)
                        obj_rot = state_check[..., 51:55].view(-1, 4)
                    else:  # use direct env
                        state_check = obs_dict['obs'].squeeze(0)
                        joint_state = obs_dict['agent_pos'].squeeze(0)
                        obj_pos = state_check[..., 48:51].view(-1, 3)
                        obj_rot = state_check[..., 51:55].view(-1, 4)
                    # 其实就是在predict action之前使用rfu来回放上一帧的轨迹然后拿到那些 distance 信息
                    this_pc = obs_dict['point_cloud'].squeeze(0)
                    obs_rfu = self.env_rfu.step(obj_pos, obj_rot, joint_state, this_pc)

                    obs_dict_input["hand_contact_point_all"] = obs_rfu['hand_contact_point_all'].unsqueeze(0)

                    # get the contact and not contact pc_xyz
                    hand_contact_point_all = obs_rfu['hand_contact_point_all']

                    # # for debug
                    # hand_contact_point_all_xyz = hand_contact_point_all[..., :3].squeeze(0)[-1]
                    # hand_contact_point_all_read = hand_contact_point_all[..., 3].squeeze(0)[-1]
                    # color_tensor = torch.zeros_like(hand_contact_point_all_xyz, dtype=torch.float32, device=device)
                    # color_tensor_indices = torch.nonzero(hand_contact_point_all_read == 1)
                    # color_tensor[color_tensor_indices] = torch.tensor([1, 0, 0], dtype=torch.float32, device=device)
                    # pc_all_debug_xyz = np.concatenate([pc_debug[:, :3], hand_contact_point_all_xyz.cpu().numpy()], axis=0)
                    # pc_all_debug_rgb = np.concatenate([pc_debug[:, 3:] / 255.0, color_tensor.cpu().numpy()], axis=0)

                    # o3d_pc.points = o3d.utility.Vector3dVector(pc_all_debug_xyz)
                    # o3d_pc.colors = o3d.utility.Vector3dVector(pc_all_debug_rgb)
                    # o3d.visualization.draw_geometries([o3d_pc])
                    # pc_all_debug_xyz = np.concatenate([hand_contact_point_all_xyz.cpu().numpy()], axis=0)
                    # pc_all_debug_rgb = np.concatenate([color_tensor.cpu().numpy()], axis=0)

                    # o3d_pc.points = o3d.utility.Vector3dVector(pc_all_debug_xyz)
                    # o3d_pc.colors = o3d.utility.Vector3dVector(pc_all_debug_rgb)
                    # o3d.visualization.draw_geometries([o3d_pc])

                    '''
                    key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
                    key: agent_pos, value.shape: torch.Size([1, 2, 24])
                    '''
                    action_dict = policy.predict_action(obs_dict_input)

                torch_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu')) # isaaclab is torch_based
                action = torch_action_dict['action'].squeeze(0).unsqueeze(1)

                obs_dict, reward, done, extras = env.step(action)

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
                
                all_traj_rewards.append(traj_reward)

            if episode_idx % 3 == 0:  # record vedio every 2 episodes
            
                videos = env.env.get_video()
                cprint(f"record vedio in episode: {episode_idx}, video.shape: {videos.shape}", "cyan") #(T, C, H, W)
                if len(videos.shape) == 5:
                    videos = videos[:, 0]  # select first frame
                
                if save_video:
                    all_videos.append(videos)
                
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
            videos_wandb = wandb.Video(all_videos, fps=self.fps, format="mp4")

            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        loss_arr = np.array(loss_list)

        return log_data
    
    

    def close_simulation_app(self): 
        self.simulation_app.close()

