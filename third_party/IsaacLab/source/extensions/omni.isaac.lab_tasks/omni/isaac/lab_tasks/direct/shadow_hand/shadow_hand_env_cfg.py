# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from omni.isaac.lab_assets.shadow_hand import SHADOW_HAND_CFG, SHADOW_HAND_REAL_CFG, SHADOW_HAND_ALIGNED_CFG, SHADOW_HAND_COLORED_CFG

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from omni.isaac.lab.sensors import ContactSensorCfg
import os


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            # "restitution_range": (1.0, 1.0),
            # "restitution_range": (0.15, 0.15),
            "restitution_range": (0.3, 0.3),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            # "restitution_range": (1.0, 1.0),
            "restitution_range": (0.3, 0.3),
            # "restitution_range": (0.15, 0.15),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

OBJ_ATTR_DICT = {


    "ring":{
        "color": None,  # magenta
        "scale": (0.3, 0.3, 0.3),
        "moving_avg": 0.3,
    },
    "vase":{
        "color": None,
        "scale": (0.08, 0.08, 0.08),
        # "scale": (0.06, 0.06, 0.06),
        "moving_avg": 0.3,
    },
    "smallvase":{
        "color": None,
        # "scale": (0.08, 0.08, 0.08),
        "scale": (0.06, 0.06, 0.06),
        "moving_avg": 0.3,
    },

    "pyramid":{
        "color": None,
        "scale": (0.155, 0.155,  0.155),
        "moving_avg": 0.3,
    },
    "apple":{
        "color": None,
        "scale": (0.16, 0.16,  0.16),
        "moving_avg": 0.3,
    },


}


@configclass
class ShadowHandEnvCfg(DirectRLEnvCfg):
    # env
    fix_wrist = False
    decimation = 2
    episode_length_s = 10.0
    num_actions = 20
    num_observations = 157  # (full)
    num_states = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = 0
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0   


@configclass
class ShadowHandOpenAIEnvCfg(ShadowHandEnvCfg):

    fix_wrist = False

    # env
    decimation = 3
    # episode_length_s = 8.0
    num_actions = 20
    num_observations = 42
    num_states = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.3
    # success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )



    # object_name = "apple"
    # object_name = "cube"
    # object_name = "vase"
    object_name = "ring"

    if object_name == "cube":
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 8.0

    elif object_name in ["ring", "vase",  "pyramid", "apple",  "smallvase"]:
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        # goal_diffuse_color = (0.0, 1.0, 0.0)  # green
        goal_diffuse_color = _diffuse_color
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        # episode_length_s = 20.0
        episode_length_s = 8.0

    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = _diffuse_color
        act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/"
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0
        if use_heavy:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model_instanceable.usd")
        else:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
    

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=_object_scale,
                visual_material=goal_visual_material,  
            )
        },
    )

    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
            visual_material=goal_visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.68), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*", history_length=2, debug_vis=contact_debug_vis
    )

@configclass
class ShadowHandRealEnvCfg(ShadowHandOpenAIEnvCfg):
    fix_wrist = True
    # fix_wrist = False

    decimation = 4

    fall_dist = 0.24
    num_actions = 24
    act_moving_average = 1.0
    action_interpolation = 0.4
    # action_interpolation = 0.25
    # action_interpolation = 1.0
    height_vis_obj = 50.0
    # num_actions = 8
    num_observations = 42 + 4
    num_states = 187 + 4
    actuated_joint_names = [
        "rh_WRJ2",
        "rh_WRJ1",
        "rh_FFJ4",
        "rh_FFJ3",
        "rh_FFJ2",
        "rh_FFJ1",
        "rh_MFJ4",
        "rh_MFJ3",
        "rh_MFJ2",
        "rh_MFJ1",
        "rh_RFJ4",
        "rh_RFJ3",
        "rh_RFJ2",
        "rh_RFJ1",
        "rh_LFJ5",
        "rh_LFJ4",
        "rh_LFJ3",
        "rh_LFJ2",
        "rh_LFJ1",
        "rh_THJ5",
        "rh_THJ4",
        "rh_THJ3",
        "rh_THJ2",
        "rh_THJ1",
    ]
    fingertip_body_names = [
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
        "rh_thdistal",
    ]
    mimic_joint_names = [
        "rh_FFJ1",
        "rh_MFJ1",
        "rh_RFJ1",
        "rh_LFJ1",
    ]
    to_mimic_joint_names = [
        "rh_FFJ2",
        "rh_MFJ2",
        "rh_RFJ2",
        "rh_LFJ2",
    ]
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # bounce_threshold_velocity=0.0,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    robot_cfg: ArticulationCfg = SHADOW_HAND_COLORED_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    # robot_cfg: ArticulationCfg = SHADOW_HAND_REAL_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.0, 0.0, -0.7071, 0.7071),
            joint_pos={".*": 0.0},
        )
    )
    # events: EventCfg = EventCfg()

    # object_name = "cube"
    # object_name = "vase"
    # object_name = "smallvase"
    object_name = "apple"
    # object_name = "ring"
    root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/"
    if object_name == "cube":
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/Blocks/DexCube/dex_cube_instanceable.usd"
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 8.0

    elif object_name in ["ring", "vase",  "pyramid", "apple", "stick", "smallvase"]:
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        # goal_diffuse_color = (0.0, 1.0, 0.0)  # green
        goal_diffuse_color = _diffuse_color
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0


    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = _diffuse_color
        # act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0
        if use_heavy:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model_instanceable.usd")
        else:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),

            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.35, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=_object_scale,
                visual_material=goal_visual_material,  
            )
        },
    )

    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=_object_scale,
            visual_material=goal_visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.68), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_forces_cfg = ContactSensorCfg(

        prim_path="/World/envs/env_.*/Robot/rh_(?!hand_mount|forearm).*", history_length=2, debug_vis=False
    )

@configclass
class ShadowHandRealHandInitEnvCfg(ShadowHandRealEnvCfg):

    # object_name = "cube"
    # object_name = "vase"
    # object_name = "smallvase"
    object_name = "apple"
    # object_name = "ring"
    root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/"

    if object_name == "cube":
        # usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/Blocks/DexCube/dex_cube_instanceable.usd"
        print(f"usd_path: {usd_path}")
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = True
        # contact_debug_vis = False
        episode_length_s = 8.0

    elif object_name in ["ring", "vase", "cup", "A", "pyramid", "apple", "stick", "smallvase"]:
        # usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/mjcf/pen_only/DAPG_pen_only.usd"
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        # goal_diffuse_color = (0.0, 1.0, 0.0)  # green
        goal_diffuse_color = _diffuse_color
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0
        # episode_length_s = 8.0

    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = _diffuse_color
        # act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0
        if use_heavy:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model_instanceable.usd")
        else:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
    

    

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                # disable_gravity=True,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                # sleep_threshold=50,
                # stabilization_threshold=250,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.35, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    if object_name != "cube":
        object_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.35, 0.555), rot=(1.0, 0.0, 0.0, 0.0))


@configclass
class ShadowHandRealHandInitPCTactileEnvCfg(ShadowHandRealHandInitEnvCfg):

    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # bounce_threshold_velocity=0.0,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    point_cloud_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="abs"),
    )

@configclass
class ShadowHandRealHandInitPCTactileSingalGoalEnvCfg(ShadowHandRealHandInitPCTactileEnvCfg): 
    pass

@configclass
class ShadowHandOpenAIFixWristEnvCfg(ShadowHandOpenAIEnvCfg):
    fix_wrist = True


@configclass
class ShadowHandOpenAIPCTactileEnvCfg(ShadowHandEnvCfg):
    # env
    fix_wirst = False
    decimation = 3
    num_actions = 20
    num_observations = 42
    num_states = 187
    asymmetric_obs = True
    obs_type = "openai"
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.0002
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.3
    # success_tolerance = 0.4
    max_consecutive_success = 50
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0
    # domain randomization config
    events: EventCfg = EventCfg()
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )


    # object_name = "cube"
    # object_name = "apple"
    # object_name = "pyramid"
    # object_name = "vase"
    object_name = "ring"

    if object_name in ["cube"]:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
        _object_scale = (1.0, 1.0, 1.0)
        visual_material = None
        goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 8.0
    
    elif object_name in ["ring", "vase", "cup", "A", "pyramid", "apple", "stick", "smallvase"]:
        # usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/mjcf/pen_only/DAPG_pen_only.usd"
        usd_path = f"/home/yijin/cfm_isaac/third_party/IsaacLab/assets/shape_variant/thingi10k/colored_obj_stl/{object_name}/usd_color/model.usd"
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = (0.0, 1.0, 0.0)  # green
        goal_diffuse_color = _diffuse_color 
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None
        contact_debug_vis = False
        episode_length_s = 20.0

    else:
        _object_scale = OBJ_ATTR_DICT[object_name]["scale"]
        _diffuse_color = OBJ_ATTR_DICT[object_name]["color"]
        goal_diffuse_color = (0.0, 1.0, 0.0)  # green

        act_moving_average = OBJ_ATTR_DICT[object_name]["moving_avg"]
        use_heavy = OBJ_ATTR_DICT[object_name]["use_heavy"]
        root_dir = "/home/yijin/cfm_isaac/third_party/IsaacLab/"
        if _diffuse_color is not None:
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=_diffuse_color)
            goal_visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=goal_diffuse_color)
        else:
            visual_material = None
            goal_visual_material = None

        contact_debug_vis = False
        episode_length_s = 20.0
        if use_heavy:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd_heavy/{object_name}/model_instanceable.usd")
        else:
            usd_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
            usd_instanceable_path = os.path.join(root_dir, f"assets/shape_variant/usd/{object_name}/model_instanceable.usd")
    

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
            visual_material=visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=_object_scale,
                visual_material=goal_visual_material,  
            )
        },
    )

    vis_goal_obj_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=_object_scale,
            visual_material=goal_visual_material,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.2, -0.45, 0.68), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_forces_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_(?!hand_mount|forearm).*", history_length=2, debug_vis=contact_debug_vis
    )

@configclass
class SHadowhandOpenAIPCTactileFixWristEnvCfg(ShadowHandOpenAIPCTactileEnvCfg):
    fix_wrist = True

@configclass
class ShadowhandOpenAIPCTactileZAxisEnvCfg(ShadowHandOpenAIPCTactileEnvCfg):
    pass


@configclass
class ShadowhandOpenAIPCTactileFixWristMultiTargetEnvCfg(SHadowhandOpenAIPCTactileFixWristEnvCfg):
    pass
