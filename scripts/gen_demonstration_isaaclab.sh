# examples:  nohup bash scripts/gen_demonstration_isaaclab.sh > out_zarr/_direct_openai_handinit_random_reset_singlecam_10000.out


cd third_party/IsaacLab

python source/standalone/workflows/rl_games/gen_expert_isaaclab_givenstep_minmem.py \
    --task Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-HandInit-PointCloud-Tactile-SingleCam-GivenStep-v0 \
    --num_envs 1  \
    --checkpoint 'logs/rl_games/shadow_hand_openai_ff/openai_ff/nn/shadow_hand_openai_ff.pth'\
    --num_point 1024 \
    --num_episodes 10000 \
    --root_dir '../../3D-Conditional-Flow-Matching/data' \
    --enable_cameras \
    --task_name cube \
    --zarr_info '_direct_openai_handinit_random_reset_singlecam_10000' \
    --record_tactile \
    --max_episode_steps 150 \
    --episode_success_threshold 30 \
    --store_every 100 \
    --backup_every 300 \
    --record_tail_steps 20 \
    --camera_numbers 1 \
    --headless \

    # --max_episode_steps 150 \
    # --episode_success_threshold 25 \

# task can be: Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-SingleGoal-v0, Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0, ... see "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/__init__.py" for more information

    # --point_cloud_debug \
    # --camera_debug \
        # --headless \
    #     --step_every 100 \