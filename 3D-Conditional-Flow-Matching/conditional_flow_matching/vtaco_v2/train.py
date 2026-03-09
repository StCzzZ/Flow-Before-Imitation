import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from networks.tracking_pipeline import *
from networks.hand_tracking_pipeline import HandPoseTrackingModule
from networks.callbacks import VisCallBack
from datasets.dataset import VTacODataModule

@hydra.main(config_path="config", config_name="train_tracking_sdf")
def main(cfg: DictConfig):
    # hydra creates working directory automatically
    print("Workding under: ", os.getcwd())
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("vis", exist_ok=True)

    if not os.path.exists(cfg.trainer.resume_from_checkpoint):
        cfg.trainer.resume_from_checkpoint = None
    
    ### Problem with DISO!!! 
    ### Use CUDA_VISIBLE_DEVICES to solve
    cfg.trainer.gpus = [0]
    
    datamodule = eval(cfg['datamodule_name'])(**cfg.datamodule)
    if "HandPose" in cfg['trackingmodule_name']:
        datamodule.setup()
        vis_seq_list = datamodule.dataset_val.datapoint_list
    else:
        vis_seq_list = None
    
    pipeline_model = eval(cfg['trackingmodule_name'])(**cfg.trackingmodule, vis_seq_list=vis_seq_list)
    
    logger = pl.loggers.TensorBoardLogger(
                save_dir=os.getcwd())

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **cfg.ckpt_param)
    
    vis_callback = VisCallBack()
    
    if "HandPose" in cfg['trackingmodule_name']:
        call_back_list = [checkpoint_callback]
    else:
        call_back_list = [checkpoint_callback, vis_callback]
    
    trainer = pl.Trainer(
        callbacks=call_back_list,
        checkpoint_callback=True,
        logger=logger, 
        **cfg.trainer)
    
    trainer.fit(model=pipeline_model, datamodule=datamodule)

# %%
# driver
if __name__ == "__main__":
    main()