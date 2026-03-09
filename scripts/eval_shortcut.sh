# use the same command as training except the script
# for example:
# bash scripts/eval_shortcut.sh cfm3d_shortcut isaaclab_cube_tactile 0308_8base 0 0 0


DEBUG=False

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed_used=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="./cfm_isaac/3D-Conditional-Flow-Matching/data/out_shortcut/${exp_name}_seed${seed_used}"

gpu_id=${5}
seed_eval=${6}


cd 3D-Conditional-Flow-Matching

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python eval_shortcut.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed_eval} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}



                                