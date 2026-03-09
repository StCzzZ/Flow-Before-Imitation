## Installation 

1. Install isaaclab according to [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

> **Note:** This work was conducted with IsaacLab version 4.1.0. You can use the latest version of IsaacLab, and we have retained the relevant IsaacLab scripts from that time in the codebase for reference.

2. Install cfm3d

    ```cd 3D-Conditional-Flow-Matching && pip install -e . && cd ..```

3. install some necessary packages

    ```pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor chamferdist open3d huggingface-hub==0.25.0 torch_scatter transforms3d```

4. Install from conditinal flow matching
    
    ```
    cd 3D-Conditional-Flow-Matching/conditional_flow_matching/conditional-flow-matching

    pip install -r requirements.txt

    cd ../..    
    ```


5. Install pyrfuniverse(rfu) from https://github.com/robotflow-initiative/rfuniverse

prepare for rich contact (face up) rfu environment: 
```
chmod +x third_party/IsaacLab/replay_new/Build_LoadMesh_Linux/shadow.x86_64 
./third_party/IsaacLab/replay_new/Build_LoadMesh_Linux/shadow.x86_64 
```


## Err Catch

1. ModuleNotFoundError: No module named 'pytorch3d': comment out all 2 pytorch3d imports
2. ImportError: cannot import name 'cached_download' from 'huggingface_hub' (.../miniconda3/envs/CFM3D/lib/python3.10/site-packages/huggingface_hub/__init__.py): pip install huggingface-hub==0.25.0



## Getting Started

Before generating data, you need to unzip the assets for the objects under `third_party/IsaacLab/assets/` and adjust your file path in `third_party/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/shadow_hand_env_cfg.py` 

1. You need to train an RL expert for the task first: 
    ```
    cd third_party/Isaaclab/
    bash scripts/run_train.sh
    ```

2. Use the RL expert to generate demostration data: 
    ```
    bash scripts/gen_demonstration_isaaclab.sh
    ```
    Generating 5K trajectories takes approximately 10 hours.

3. Replay Demonstration Data: after configuring the rfuniverse environment and adjusting the file paths accordingly, you should be able to directly run the replay script to obtain tactile reading labels from the demonstration data.

    ```
    cd third_party/IsaacLab/
    python replay_new/run_replay_new.py
    ```


4. Train & evaluate FBI

    The bash script is at: `scripts/train_shortcut.sh`. An example of training command is:

    ```
    bash scripts/train_shortcut.sh cfm3d_shortcut_rich_contact isaaclab_cube_tactile_rich_contact 1212 0 0
    ```

    **Evaluate**: Script is at: `scripts/eval_shortcut.sh`. An example of evaluation command is:
    ```
    bash scripts/eval_shortcut.sh cfm3d_shortcut_rich_contact isaaclab_cube_tactile_rich_contact 1212 0 0
    ```
