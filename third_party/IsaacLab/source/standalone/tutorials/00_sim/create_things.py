# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    from omni.isaac.lab_assets.shadow_hand import SHADOW_HAND_CFG #  use modified version here
    from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg


    import omni.isaac.lab.envs.mdp as mdp
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.managers import EventTermCfg as EventTerm
    from omni.isaac.lab.managers import SceneEntityCfg
    from omni.isaac.lab.markers import VisualizationMarkersCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
    from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
    from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

    from dataclasses import MISSING

    

    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # rigid_body_enabled=False,
                kinematic_enabled=True,
                disable_gravity=True,
                # kinematic_enabled=False,
                # disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # ball_radius = 0.03

    # vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/goal_object",
    #     spawn=sim_utils.SphereCfg(
    #         radius=ball_radius,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=567.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=0.5,
    #             dynamic_friction=0.5,
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.73), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    vis_goal_obj_cfg.spawn.func("/World/goal_object", vis_goal_obj_cfg.spawn)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
