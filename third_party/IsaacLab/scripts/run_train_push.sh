# bash scripts/run_train_push.sh 

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/train.py \
    --task ShadowInhandPushWristFixed-v0 \
    --num_envs 10000  \
    --headless \

