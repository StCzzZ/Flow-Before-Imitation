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


class IsaaclabRealRunner(BaseRunner):
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
                # use_contact_forces = True,
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
        # self.use_contact_forces = use_contact_forces
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

        assert(self.args_cli.task in [
     "Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-v0",
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
        
        self.agent_cfg["params"]["load_checkpoint"] = True

        # create runner from rl-games
        self.runner = Runner()
        self.runner.load(self.agent_cfg)
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
            'goal_point_cloud': gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(512, 6), 
                dtype=np.float32
            ),
            
        })

        if self.args_cli.task in [
            "Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-v0",
            ]:
            env.observation_space['obs'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(42+4, )
            )
            env.observation_space['states'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(187+4, ),
            )
                    
        elif self.args_cli.task in []:
            env.observation_space['obs'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(187+4, )
            )
        else:
            raise NotImplementedError(f"[IsaaclabRunner]task: {self.args_cli.task} is not implemented yet")

        # if self.use_contact_forces:
        #     env.observation_space['contact_forces'] = gym.spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=(24, 3),
        #         dtype=np.float32
        #     )
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

        all_videos = []

        """Play with RL-Games agent."""
        # for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in IsaacLab {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
        for episode_idx in range(self.eval_episodes):
            assert(self.simulation_app.is_running())
            # start rollout
            # if episode_idx == 0:
            obs_dict = env.reset()
            policy.reset()
            # obs = self.agent.obs_to_torch(obs_dict) 
            # required: enables the flag for batched observations
            # _ = self.agent.get_batch_size(obs, 1)
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
                    obs_dict_input['goal_point_cloud'] = obs_dict['goal_point_cloud'].unsqueeze(0)

                    # pass pass
                    # if policy.use_contact_forces:
                    #     obs_dict_input['contact_forces'] = obs_dict['contact_forces'].unsqueeze(0)


                    '''
                    key: point_cloud, value.shape: torch.Size([1, 2, 1024, 6])
                    key: agent_pos, value.shape: torch.Size([1, 2, 24])
                    '''
                    action_dict = policy.predict_action(obs_dict_input)
         
                    # visualize the pc
                    point_cloud_check = obs_dict['point_cloud'][-1].squeeze(0).detach().cpu()
                    o3d_pc_check = o3d.geometry.PointCloud()
                    o3d_pc_check.points = o3d.utility.Vector3dVector(point_cloud_check[..., :3].numpy())
                    o3d_pc_check.colors = o3d.utility.Vector3dVector(point_cloud_check[..., 3:].numpy() / 255.0 )

                    goal_point_cloud_check = obs_dict['goal_point_cloud'][-1].squeeze(0).detach().cpu()
                    o3d_goal_pc_check = o3d.geometry.PointCloud()
                    o3d_goal_pc_check.points = o3d.utility.Vector3dVector(goal_point_cloud_check[..., :3].numpy())
                    o3d_goal_pc_check.colors = o3d.utility.Vector3dVector(goal_point_cloud_check[..., 3:].numpy() / 255.0 )

                    # if (local_step % 10) == 0:
                    #     o3d.visualization.draw_geometries([o3d_pc_check, o3d_goal_pc_check])
                
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

        return log_data
    

    def close_simulation_app(self): 
        self.simulation_app.close()



