# bash scripts/run_train.sh 



TORCH_USE_CUDA_DSA=1 HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-Real-v0 \
    --num_envs 5000 \
    --headless \

