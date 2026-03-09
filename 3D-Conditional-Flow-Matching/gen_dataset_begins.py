if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from conditional_flow_matching.policy.cfm3d import CFM3D
from conditional_flow_matching.dataset.base_dataset import BaseDataset
from conditional_flow_matching.env_runner.base_runner import BaseRunner
from conditional_flow_matching.common.checkpoint_util import TopKCheckpointManager
from conditional_flow_matching.common.pytorch_util import dict_apply, optimizer_to
from conditional_flow_matching.model.conditional_unet.ema_model import EMAModel
from conditional_flow_matching.model.common.lr_scheduler import get_scheduler
from conditional_flow_matching.common.replay_buffer import ReplayBuffer
import open3d as o3d
# ignore the warnings
import warnings
warnings.filterwarnings("ignore")

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainCFM3DWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        cprint(f"Output directory: {self.output_dir}", 'magenta')
        cprint(f"learning_rate: {cfg.optimizer.lr}", 'magenta')
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: CFM3D = hydra.utils.instantiate(cfg.policy)

        self.ema_model: CFM3D = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    cprint(f"raw_loss_cpu: {raw_loss_cpu}", "yellow")
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        '''
                        wandb: WARNING Step only supports monotonically increasing values ...
                        related solutions: https://community.wandb.ai/t/warning-user-provided-step-is-less-than-current-step/5968
                        This do not affect training yet affects the logging process
                        '''
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # log all
                step_log.update(runner_log)

            
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    

                    pred_action = result['action_pred']                
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log
        
        # end of training
        if env_runner.simulation_app != None:
            cprint(f"Closing simulation app", 'magenta')
            env_runner.env.env.env.close()
            env_runner.simulation_app.close()

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        # lastest_ckpt_path = self.get_checkpoint_path(tag="best")
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest") # changed
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run_with_start_pose(policy, start_pos=self.start_pos_arr, start_rot=self.start_rot_arr, save_video= True)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
        # end of training
        if env_runner.simulation_app != None:
            cprint(f"Closing simulation app", 'magenta')
            env_runner.env.env.env.close()
            env_runner.simulation_app.close()
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

    def checkdataset(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        

        # check
        data_buffer: ReplayBuffer = dataset.replay_buffer           

        # state 是 gt, 
            
        state = data_buffer['state']
        episode_ends = data_buffer.root['meta']['episode_ends']
        pc_all = data_buffer['point_cloud']

        episode_start = [0] + episode_ends.tolist()[:-1]
        cprint(f"episode_start: {episode_start}", 'green')

        start_pos_list = []
        start_rot_list = []

        goal_rot_list = []


        for i in episode_start:
            obj_v = state[i][..., 55:58]
            obj_v_rot = state[i][..., 58:62]
            assert(np.all(obj_v_rot == 0) and np.all(obj_v == 0))
            
            # check if the last state is correct
            if i > 0:
                obj_pos_last = state[i-1][..., 48:51]
                obj_rot_last = state[i-1][..., 51:55]
                cprint(f"i: {i}, obj_pos_last: {obj_pos_last}, obj_rot_last: {obj_rot_last}", 'magenta')
                pc_last = pc_all[i-1]
                pcd_last = o3d.geometry.PointCloud()
                pcd_last.points = o3d.utility.Vector3dVector(pc_last[..., :3])
                pcd_last.colors = o3d.utility.Vector3dVector(pc_last[..., 3:6] / 255.0)
                # o3d.visualization.draw_geometries([pcd_last])

            pc_next = pc_all[i + 1]
            pcd_next = o3d.geometry.PointCloud()
            pcd_next.points = o3d.utility.Vector3dVector(pc_next[..., :3])
            pcd_next.colors = o3d.utility.Vector3dVector(pc_next[..., 3:6] / 255.0)
            # o3d.visualization.draw_geometries([pcd_next])
            # 此时查看位置，就是需要的初始位置

            # np.squeeze() 不会自动检查的，如果是1的话会raise err
            obj_pos = state[i][..., 48:51]
            obj_rot = state[i][..., 51:55]
            goal_rot = state[i][..., 64:68]
            cprint(f"i: {i}, obj_pos: {obj_pos}, obj_rot: {obj_rot}, goal_rot: {goal_rot}", 'yellow')

            # check whether the end of the last obj_rot is close to goal_rot
            # if i > 0:
            #     obj_rot_last = state[i-1][..., 51:55]

            

            start_pos_list.append(torch.tensor(obj_pos, dtype=torch.float32, device=self.model.device))
            start_rot_list.append(torch.tensor(obj_rot, dtype=torch.float32, device=self.model.device))
            goal_rot_list.append(torch.tensor(goal_rot, dtype=torch.float32, device=self.model.device))

            pc_i = pc_all[i]

            #  visualize using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_i[..., :3])
            pcd.colors = o3d.utility.Vector3dVector(pc_i[..., 3:6] / 255.0)
            # o3d.visualization.draw_geometries([pcd])

            

        
        self.start_pos_arr = torch.stack(start_pos_list, axis=0)
        self.start_rot_arr = torch.stack(start_rot_list, axis=0)
        self.goal_rot_arr = torch.stack(goal_rot_list, axis=0)

        cprint(f"start_pos_arr: {self.start_pos_arr}, \n start_rot_arr: {self.start_rot_arr}", 'yellow')
        cprint(f"goal_rot_arr: {self.goal_rot_arr}", 'yellow')
        


        # #############################
        # # data prepare ends
        # #############################

        # #############################
        # # start evaluating 
        # #############################
        # # load the latest checkpoint
        
        # cfg = copy.deepcopy(self.cfg)
        
        # # lastest_ckpt_path = self.get_checkpoint_path(tag="best")
        # lastest_ckpt_path = self.get_checkpoint_path(tag="latest") # changed
        # if lastest_ckpt_path.is_file():
        #     self.load_checkpoint(path=lastest_ckpt_path)
        
        # # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseRunner)
        # policy = self.model
        # if cfg.training.use_ema:
        #     policy = self.ema_model
        # policy.eval()
        # policy.cuda()

        # runner_log = env_runner.run_with_start_pose(policy, start_pos=self.start_pos_arr, start_rot=self.start_rot_arr, save_video= True)
        # for key, value in runner_log.items():
        #     if isinstance(value, float):
        
        # # end of training
        # if env_runner.simulation_app != None:
        #     env_runner.env.env.env.close()
        #     env_runner.simulation_app.close()

        





        # 没问题的，确实是和[1, 0, 0, 0]相近
        # episode_ends = episode_ends - 1

        # for i in episode_ends:
        #     obj_v = state[i][..., 55:58]
        #     obj_v_rot = state[i][..., 58:62]
        #     # assert(np.all(obj_v_rot == 0) and np.all(obj_v == 0))
        #     # 此时查看位置，就是需要的初始位置

        #     # np.squeeze() 不会自动检查的，如果是1的话会raise err
        #     obj_pos = state[i][..., 48:51]
        #     obj_rot = state[i][..., 51:55]


    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'conditional_flow_matching', 'config'))
)
def main(cfg):
    workspace = TrainCFM3DWorkspace(cfg)
    workspace.checkdataset()

if __name__ == "__main__":
    main()

    
    
