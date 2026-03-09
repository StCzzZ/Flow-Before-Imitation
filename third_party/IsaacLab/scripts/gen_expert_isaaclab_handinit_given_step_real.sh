# bash scripts/gen_expert_isaaclab_handinit_given_step_real.sh 

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem_real.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-Direct-Real-HandInit-PointCloud-Tactile-GivenStep-SingleGoal-v0\
    --num_envs 1  \
    --checkpoint '/home/yijin/cfm_isaac/third_party/IsaacLab/logs/rl_games/shadow_hand_openai_ff/apple/2025-02-05_17-49-58/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 500 \
    --root_dir '../../3D-Conditional-Flow-Matching/data/' \
    --enable_cameras \
    --task_name apple_single_goal \
    --zarr_info 'inhand_reorient_apple_single_goal_decimation4_inter0.4_bounce_fixed' \
    --record_tactile \
    --max_episode_steps 125 \
    --episode_success_threshold 2 \
    --store_every 250 \
    --backup_every 500  \
    --record_tail_steps 30 \
    --camera_numbers 1 \
    --headless \
    # --record_rgba \
    # --point_cloud_debug \
    # --headless \
    # --camera_debug \
