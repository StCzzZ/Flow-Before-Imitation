"""Script to play a checkpoint if an RL agent from RL-Games, add the pc implement by STCZZZ"""

"""Launch Isaac Sim Simulator first."""

import argparse
# import os
# import sys
# from termcolor import  cprint
# IsaacLab_path = os.path.join(os.path.abspath(os.path.curdir), "third_party/IsaacLab")
# # cprint(f"IsaacLab_path: {IsaacLab_path}", "cyan")
# sys.path.append(IsaacLab_path)


# from omni.isaac.lab_tasks.direct.allegro_hand import allegro_hand_env_cfg

from omni.isaac.lab.app import AppLauncher

# /home/yijin/cfm_isaac/third_party/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/allegro_hand/allegro_hand_env_cfg.py
# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="/home/yijin/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/2024-08-23_16-49-54/nn/shadow_hand_openai_ff.pth", help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)


# add params
parser.add_argument("--num_points", type=int, default=512, help="Number of Points to crop.")
parser.add_argument("--camera_debug", action="store_true", default=False, help="Enable camera debugging.")
parser.add_argument("--point_cloud_debug", action="store_true", default=False, help="Enable point cloud debugging.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play.")
parser.add_argument("--root_dir", type=str, default="3D-Conditional-Flow-Matching/data/", help="Root directory to save data.")

'''ValueError: The passed ArgParser object already has the field "headless"'''
# parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
# parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras rendering, see third_party/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/app/app_launcher.py.")

# import os
# from termcolor import cprint

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

'''check the environment name !!'''
assert(args_cli.task in ["Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-PointCloud-v0"])

"""Rest everything follows."""
import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# add imports
from termcolor import cprint
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd
import matplotlib.pyplot as plt
import open3d as o3d
import copy
import zarr
import numpy as np



''' define these functions and class here for convenience, and maybe we can move them to a separate file later'''
class PointcloudVisualizer() :
	def __init__(self) -> None:
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()
		# self.vis.register_key_callback(key, your_update_function)
	
	def add_geometry(self, cloud) :
		self.vis.add_geometry(cloud)

	def update(self, cloud):
		#Your update routine
		self.vis.update_geometry(cloud)
		self.vis.update_renderer()
		self.vis.poll_events()



def farthest_point_sampling(point_cloud, num_points):

        sampled_points = o3d.geometry.PointCloud.farthest_point_down_sample(point_cloud, num_points)


        return sampled_points


def get_pc_and_color(obs, env_id, camera_numbers):
    points_all = []
    colors_all = []
    for cam_id in range(camera_numbers):
        rgba_all = obs.get(f"rgba_img_0{cam_id}", None)
        depth_all = obs.get(f"depth_img_0{cam_id}", None)
        intrinsic_matrices_all = obs.get(f"intrinsic_matrices_0{cam_id}", None)
        pos_w_all = obs.get(f"pos_w_0{cam_id}", None)
        quat_w_ros_all = obs.get(f"quat_w_ros_0{cam_id}", None)

        rgba = rgba_all[env_id]
        depth = depth_all[env_id]
        intrinsic_matrix = intrinsic_matrices_all[env_id]
        pos_w = pos_w_all[env_id]
        quat_w_ros = quat_w_ros_all[env_id]

        # generate point cloud
        points_xyz, points_rgb = create_pointcloud_from_rgbd(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            rgb=rgba,
            normalize_rgb=True,  # normalize to get 0~1 pc, the same as dp3
            position=pos_w,
            orientation=quat_w_ros,
        )

        # add points and colors to list
        points_all.append(points_xyz)
        colors_all.append(points_rgb)

    # concatenate points and colors
    points_all = torch.cat(points_all, dim=0)
    colors_all = torch.cat(colors_all, dim=0)

    return points_all, colors_all



def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]) 
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    cprint(f"env_cfg: {env_cfg}", "cyan")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode = None)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs_torch = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs_torch, 1)
    
    
    ''' Modified to visualize PC'''

    # I kept the two divisions the same in order to avoid bugs while emphasising the control of "PC_debug" here
    if args_cli.point_cloud_debug:
        # initialize point cloud visualizer
        pointCloudVisualizer = PointcloudVisualizer()
        pointCloudVisualizerInitialized = False
        o3d_pc = o3d.geometry.PointCloud()
        o3d_sampled = o3d.geometry.PointCloud()
    else:   
        # pointCloudVisualizer = PointcloudVisualizer()
        # pointCloudVisualizerInitialized = False
        o3d_pc = o3d.geometry.PointCloud()
        o3d_sampled = o3d.geometry.PointCloud()
    
    # initialize the camera_numbers
    camera_numbers = 2

    if args_cli.camera_debug:
        fig, axes = plt.subplots(camera_numbers, 2)   # axes shape: (camera_numbers, 2)

    # initialize the pointcloud, state, action... buffer here. We only implement a one-env version here. 
    pointcloud_list = []
    state_list = []
    obs_list = []
    action_list = []
    image_list = []
    depth_list = []
    episode_ends_list = []
    global_steps = 0
    episode_cnt = 0
    num_episodes = args_cli.num_episodes
    success = False

    pointcloud_list_sub = []
    state_list_sub = []
    obs_list_sub = []
    action_list_sub = []
    image_list_sub = []
    depth_list_sub = []

    # main loop
    while simulation_app.is_running() and episode_cnt < num_episodes:
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs_torch = agent.obs_to_torch(obs)  

            # for key, val in obs.items():

            '''
                device: cuda:0 for all
                obs[obs].shape: torch.Size([1, 42])
                obs[states].shape: torch.Size([1, 187])
                obs[policy].shape: torch.Size([1, 42])
                obs[critic].shape: torch.Size([1, 187])
                obs[rgba_img_00].shape: torch.Size([1, 480, 640, 4])
                obs[depth_img_00].shape: torch.Size([1, 480, 640])
                obs[intrinsic_matrices_00].shape: torch.Size([1, 3, 3])
                obs[pos_w_00].shape: torch.Size([1, 3])
                obs[quat_w_ros_00].shape: torch.Size([1, 4])
                obs[rgba_img_01].shape: torch.Size([1, 480, 640, 4])
                obs[depth_img_01].shape: torch.Size([1, 480, 640])
                obs[intrinsic_matrices_01].shape: torch.Size([1, 3, 3])
                obs[pos_w_01].shape: torch.Size([1, 3])
                obs[quat_w_ros_01].shape: torch.Size([1, 4])
                obs[goal_env_ids].shape: torch.Size([0])
            '''

            # we need obs from the last step and actions predicted by the agent to do the imitation learning
            # record imgs
            for cam_idx in range(camera_numbers):
                obs_img = obs.get(f"rgba_img_0{cam_idx}", None)
                obs_depth = obs.get(f"depth_img_0{cam_idx}", None)
                image_list_sub.append(obs_img.squeeze(0).detach().cpu().numpy())
                depth_list_sub.append(obs_depth.squeeze(0).detach().cpu().numpy())

            state_list_sub.append(obs['states'].squeeze(0).detach().cpu().numpy())
            obs_list_sub.append(obs['obs'].squeeze(0).detach().cpu().numpy())
            
            ''' The PC part, add by STCZZZ'''
            
            # generate pc and collect it into the pointcloud_list_sub
            for env_id in range(args_cli.num_envs):
                points_all, colors_all = get_pc_and_color(obs, env_id, camera_numbers)
                points_env = o3d.geometry.PointCloud()
                points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
                points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())
                # farthest point sampling
                points_env = farthest_point_sampling(points_env, args_cli.num_points)
                # combine points and colors
                combined_points_colors = np.concatenate([np.asarray(points_env.points), np.asarray(points_env.colors)], axis=1)
                # points_colors_all = torch.cat([torch.tensor(points_env.points), torch.tensor(points_env.colors)], dim=1)
                points_colors_all = torch.tensor(combined_points_colors)
                pointcloud_list_sub.append(points_colors_all.squeeze(0).detach().cpu().numpy())

            # visualize image and pc if neede
            if args_cli.camera_debug:
                
                for cam_id in range(camera_numbers):
                    # visualize the depth img
                    depth_example = obs[f"depth_img_0{cam_id}"][0]
                    rgba_example = obs[f"rgba_img_0{cam_id}"][0][..., :3]
                    ax1 = axes[cam_id][0]
                    ax2 = axes[cam_id][1]
                    ax1.imshow(depth_example.detach().cpu().numpy())
                    ax1.set_title(f'Depth Image_{cam_id}')
                    ax2.imshow(rgba_example.detach().cpu().numpy())
                    ax2.set_title(f'RGBA Image_{cam_id}')
                plt.pause(1e-9)
            
            if args_cli.point_cloud_debug:
                
                points_example, colors_example = get_pc_and_color(obs, 0, camera_numbers)

                # visualize point cloud
                points_xyz_numpy = points_example.detach().cpu().numpy()
                points_rgb_numpy = colors_example.detach().cpu().numpy()

                # change the cordinate between IsaacLab and Open3d
                selected_points = points_xyz_numpy[:, [1, 2, 0]]
                selected_points_color = points_rgb_numpy
                selected_points[:, 2] = -selected_points[:, 2]
                selected_points[:, 0] = -selected_points[:, 0]
            
                o3d_pc.points = o3d.utility.Vector3dVector(selected_points)
                o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color)
                o3d_pc_tmp = farthest_point_sampling(o3d_pc, args_cli.num_points)
                o3d_sampled.points, o3d_sampled.colors = o3d_pc_tmp.points, o3d_pc_tmp.colors
                                
                # visualize point cloud using open3d.
                # o3d.visualization.draw_geometries([o3d_sampled])
                if pointCloudVisualizerInitialized == False :
                    pointCloudVisualizer.add_geometry(o3d_sampled)
                    pointCloudVisualizerInitialized = True
                else :
                    pointCloudVisualizer.update(o3d_sampled)
            

            # agent stepping   modified
            actions = agent.get_action(obs_torch, is_deterministic=True)
            action_list_sub.append(actions.detach().cpu().numpy().squeeze(0))
            # env stepping
            obs, reward, dones, extras = env.step(actions)

            goal_env_ids = obs['goal_env_ids']
            success = len(goal_env_ids) > 0
            if success:
                
                cprint(f"Success in episode {episode_cnt}, reward: {reward}", "green")
                cprint(f"Episode steps: {len(action_list_sub)}", "green")

                global_steps += len(action_list_sub)
                episode_ends_list.append(global_steps)
                episode_cnt += 1

                # record the episode
                pointcloud_list.extend(copy.deepcopy(pointcloud_list_sub))
                state_list.extend(copy.deepcopy(state_list_sub))
                obs_list.extend(copy.deepcopy(obs_list_sub))
                action_list.extend(copy.deepcopy(action_list_sub))
                image_list.extend(copy.deepcopy(image_list_sub))
                depth_list.extend(copy.deepcopy(depth_list_sub))

                # reset lists
                pointcloud_list_sub = []
                state_list_sub = []
                obs_list_sub = []
                action_list_sub = []
                image_list_sub = []
                depth_list_sub = []
            # perform operations for terminated episodes
            
            if dones[0] : # timeout or failed

                # reset lists
                pointcloud_list_sub = []
                state_list_sub = []
                obs_list_sub = []
                action_list_sub = []
                image_list_sub = []
                depth_list_sub = []
    
    # close the simulator
    env.close()

    cprint(f"saving data to zarr file...", "cyan")

    # save all datas
    assert(episode_cnt == num_episodes)
    # save data
    ###############################
    # save data
    ###############################

    # check if save_dir exists
    save_dir = os.path.join(args_cli.root_dir, 'isaaclab_'+ args_cli.task +'_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return

    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    

    # no need to do, the channel is already last
    # if img_arrays.shape[1] == 3: # make channel last
    #     img_arrays = np.transpose(img_arrays, (0,2,3,1))

    



    img_arrays = np.stack(image_list, axis=0)
    state_arrays = np.stack(state_list, axis=0)
    obs_arrays = np.stack(obs_list, axis=0)
    point_cloud_arrays = np.stack(pointcloud_list, axis=0)
    depth_arrays = np.stack(depth_list, axis=0)
    action_arrays = np.stack(action_list, axis=0)
    episode_ends_arrays = np.array(episode_ends_list)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    obs_size = (100, obs_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('obs', data=obs_arrays, chunks=obs_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    cprint(f'-'*50, 'cyan')
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'obs shape: {obs_arrays.shape}, range: [{np.min(obs_arrays)}, {np.max(obs_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()