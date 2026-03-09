# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/add_sensors_on_robot.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
    )


# add by STCZZZ
import numpy as np
import open3d as o3d
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd

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


def rand_row(tensor, color_tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:], color_tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

def sample_points(points, points_color, sample_num=1000, sample_mathed='random'):
    # eff_points = points[points[:, 2]>0.04]
    # if eff_points.shape[0] < sample_num :
    #     eff_points = points
    if sample_mathed == 'random':
        sampled_points, sampled_points_color = rand_row(points, points_color, sample_num)
    return sampled_points, sampled_points_color


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # define the pc params
    pointCloudVisualizer = PointcloudVisualizer()
    pointCloudVisualizerInitialized = False
    o3d_pc = o3d.geometry.PointCloud()
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("pos_w.shape: ", scene["camera"].data.pos_w.shape)
        print("quat_w_ros shape: ", scene["camera"].data.quat_w_ros.shape)
        print("-------------------------------")
        print(scene["height_scanner"])
        print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        print("-------------------------------")
        print(scene["contact_forces"])
        print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())

        '''
        Received shape of rgb   image:  torch.Size([2, 480, 640, 4])
        Received shape of depth image:  torch.Size([2, 480, 640])
        pos_w: Shape is (N, 3), where N is the number of sensors.(from tutorial https://isaac-sim.github.io/IsaacLab/source/api/lab/omni.isaac.lab.sensors.html)
        '''
        
        # visualize the cam data
        camera_debug = True
        if camera_debug:
            import matplotlib.pyplot as plt
            # camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")

            # 一共有两个相机，这里查看第一个
            camera_depth_img = scene["camera"].data.output["distance_to_image_plane"].detach().cpu().numpy()[1]
            # camera_rgb_image = camera_rgba_image[..., :3]
            plt.imshow(camera_depth_img)
            plt.pause(1e-9)

            # camera_depth_img = scene["camera"].data.output["distance_to_image_plane"].detach().cpu().numpy()[1]
            # # camera_rgb_image = camera_rgba_image[..., :3]
            # plt.imshow(camera_depth_img)
            # plt.pause(1e-9)

        
        # Pointcloud in world frame, reference: https://isaac-sim.github.io/IsaacLab/source/how-to/save_camera_output.html
        points_xyz, points_rgb = create_pointcloud_from_rgbd(
             intrinsic_matrix=scene["camera"].data.intrinsic_matrices[1],
             depth=scene["camera"].data.output["distance_to_image_plane"][1],
             rgb=scene["camera"].data.output["rgb"][1],
             normalize_rgb=True,
             position=scene["camera"].data.pos_w[1],
             orientation=scene["camera"].data.quat_w_ros[1],
             device=sim.device,
        )

        # here the "rgb" can be either "rgb" or "rgba" image

        pointcloud_debug = True
        
        import einops
        if pointcloud_debug:
            # if pointCloudVisualizer != None :
            # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
            # convert them directly will cause RuntimeError: Unable to cast Python instance of type <class 'torch.Tensor'> to C++ type
            # so we need to convert them to numpy first
            # points_rgb_numpy = points_rgb.detach().cpu().numpy()
            
            points_xyz_numpy = points_xyz.detach().cpu().numpy()
            points_rgb_numpy = points_rgb.detach().cpu().numpy()
            # selected_points, selected_points_color =  sample_points(points_xyz_numpy, points_rgb_numpy, sample_num=1024, sample_mathed='random')
            
            # selected_points = einops.rearrange(points_xyz_numpy, "N,(x, y, z) -> N, (z, x, y)") 
            # selected_points_color = einops.rearrange(points_rgb_numpy, "x, y, z -> z, x, y")

            # selected_points = points_xyz_numpy
            # selected_points_color = points_rgb_numpy

            selected_points = points_xyz_numpy[:, [1, 2, 0]]
            selected_points_color = points_rgb_numpy
            # selected_points_color = points_rgb_numpy[:, [1, 2, 0]]  # 颜色不用换，颜色的三个代表RGB，颜色是point_wise 的

            # selected_points[:, 1] 和 selected_points[:, 2] 的值都设置成原来的相反数
            # selected_points[:, 1] = -selected_points[:, 1]
            selected_points[:, 2] = -selected_points[:, 2]
            selected_points[:, 0] = -selected_points[:, 0]
            
            print(f"selected_points.shape: {selected_points.shape}")
            print(f"selected_points_color.shape: {selected_points_color.shape}")
            print(f"selected_points: {selected_points.dtype}")
            print(f"selected_points_color: {selected_points_color.dtype}")
            o3d_pc.points = o3d.utility.Vector3dVector(selected_points)

            # 先不设置color
            # o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color / 256.)
            # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])
            o3d_pc.colors = o3d.utility.Vector3dVector(selected_points_color)

            
            # o3d.visualization.draw_geometries([o3d_pc, FOR1])
            if pointCloudVisualizerInitialized == False :
                pointCloudVisualizer.add_geometry(o3d_pc)
                # pointCloudVisualizer.add_geometry(FOR1)
                pointCloudVisualizerInitialized = True
            else :
                pointCloudVisualizer.update(o3d_pc)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
