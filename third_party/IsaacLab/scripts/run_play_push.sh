# bash scripts/run_train_push.sh 

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python source/standalone/workflows/rl_games/play.py \
    --task ShadowInhandPushWristFixed-v0 \
    --num_envs 1   \
    --checkpoint 'your ckpt path' \

    
