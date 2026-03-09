

# nohup bash scripts/train_shortcut_real.sh cfm3d_shortcut_real isaaclab_cube_real 0130_noise 0 0 > ./out_shortcut_real/0130_noise.out



# nohup bash scripts/train_shortcut_real.sh cfm3d_shortcut_real isaaclab_cube_real 0209_apple_noise 0 0 > ./out_shortcut_real/0209_apple_noise.out

# nohup bash scripts/train_shortcut_real.sh cfm3d_shortcut_real isaaclab_cube_real 0210_smallvase_noise 0 0 > ./out_shortcut_real/0210_smallvase_noise.out

# nohup bash scripts/train_shortcut_real.sh cfm3d_shortcut_real isaaclab_cube_real 0211_vase_noise 0 0 > ./out_shortcut_real/0211_vase_noise.out





# cfm3d = alg_name, adroit_hammer=taskname, 0322=additioninfo, 0332 is just for name, no other uses. 
# 0=seed, 0=gpuid


mkdir -p out_shortcut_real

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./cfm_isaac/3D-Conditional-Flow-Matching/data/out_shortcut/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Conditional-Flow-Matching


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_shortcut_real.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                