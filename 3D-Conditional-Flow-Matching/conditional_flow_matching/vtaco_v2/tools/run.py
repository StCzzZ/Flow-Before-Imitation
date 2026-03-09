import os
import yaml
import argparse
import subprocess, os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run train')
    parser.add_argument("--data", type=str, default="vtaco")
    parser.add_argument("--type", type=str, default="sdf")
    parser.add_argument("--obj", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()
    
    gpu = args.gpu
    
    config_path = "config/{}".format(args.data)
    
    if args.type == "sdf":
        config_name = "tracking_sdf_{:03d}".format(args.obj)
        sh_name = "sdf_{:03d}".format(args.obj)
    if args.type == "hand":
        config_name = "hand_pose_tracking_{:03d}".format(args.obj)
        sh_name = "hand_pose_{:03d}".format(args.obj)
    if args.type == "geo":
        config_name = "geo_{:03d}".format(args.obj)
        sh_name = "geo_{:03d}".format(args.obj)

    
    ### Load GPU id from config
    if gpu == -1:
        with open("{}/{}.yaml".format(config_path, config_name), "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            gpu = cfg['trainer']['gpus'][0]
    
    run_cmd = "CUDA_VISIBLE_DEVICES={} bash ~/vtaco_v2/tools/{}/{}.sh".format(gpu, args.data, sh_name)
    
    os.system(run_cmd)
    
