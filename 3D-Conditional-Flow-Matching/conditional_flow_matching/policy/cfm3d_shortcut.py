from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from conditional_flow_matching.torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from termcolor import cprint
import copy
import time
# import pytorch3d.ops as torch3d_ops


from conditional_flow_matching.model.common.normalizer import LinearNormalizer
from conditional_flow_matching.policy.base_policy import BasePolicy
# from conditional_flow_matching.model.conditional_unet.conditional_unet1d import ConditionalUnet1D
from conditional_flow_matching.model.conditional_unet.conditional_unet1d_shortcut import ConditionalUnet1DShortCut
from conditional_flow_matching.model.conditional_unet.mask_generator import LowdimMaskGenerator
from conditional_flow_matching.common.pytorch_util import dict_apply, dict_apply_print
from conditional_flow_matching.common.model_util import print_params
from conditional_flow_matching.model.vision.pointnet_extractor import CFMEncoder
# addition
import torchdiffeq

class CFM3D_Shortcut(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            # noise_scheduler: DDPMScheduler,
            flow_matcher: ConditionalFlowMatcher,  # the ot-trans-matcher is a child class of CFM, so it is OK
            horizon, 
            n_action_steps, 
            n_obs_steps,
            # num_inference_steps=None,
            obs_as_global_cond=True,
            flow_matching_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            use_contact_forces=False,
            encoder_type = "CFM3D", # can be CFM3D, Fused or Pointnet-Fused or Pointnet-Large
            pointcloud_encoder_cfg=None,
            # added params
            prediction_type = "conditional_flow_matching",

            # params for inference
            ode_method = "euler",
            inference_step = 10,
            use_shortcut = False,
            denoise_time_all = 128,
            bootstrap_every = 8,
            atol = 1e-4,
            rtol = 1e-4,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        if encoder_type == "CFM3D":
            obs_encoder = CFMEncoder(observation_space=obs_dict,
                                    img_crop_shape=crop_shape,
                                    out_channel=encoder_output_dim,
                                    pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                    use_pc_color=use_pc_color,
                                    pointnet_type=pointnet_type,
                                    use_contact_forces=use_contact_forces,
                                    # add by STCZZZ
                                    # state_mlp_size=(32, 32),
                                    # contact_mlp_size=(32, 32),
                                    state_mlp_size=(64, 64),
                                    contact_mlp_size=(64, 64),
                                )
        else:
            raise NotImplementedError(f"Unsupported encoder type {encoder_type}")


        # create flowmatching model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.use_contact_forces = use_contact_forces
        cprint(f"[CFMUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[CFMUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        cprint(f"[CFMUnetHybridPointcloudPolicy] use_contact_forces: {self.use_contact_forces}", "yellow")
        cprint(f"[CFM3D] obs_encoder: {obs_encoder}", "yellow")

        # model for shortcut
        model = ConditionalUnet1DShortCut(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            flow_matching_embed_dim=flow_matching_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.flow_matcher = flow_matcher
        
        
        self.flow_matcher_pc = copy.deepcopy(flow_matcher)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond

        self.prediction_type = prediction_type
        self.ode_method = ode_method
        self.inference_step = inference_step
        self.use_shortcut = use_shortcut
        self.denoise_time_all = denoise_time_all
        self.bootstrap_every = bootstrap_every
        self.atol = atol
        self.rtol = rtol

        # eps:  the The smallest time step to sample from, imitating https://github.com/gnobitab/RectifiedFlow
        # we continue to use this setting, avoid the exact "0" here
        self.T = 1.
        self.eps = 1e-5


        self.kwargs = kwargs


        print_params(self)
        cprint(f"Model_name: {self.__class__.__name__}", "magenta")
        cprint(f"flow_matcher: {flow_matcher}" , "magenta")
        cprint(f"prediction_type: {prediction_type}" , "magenta")
        cprint(f"inference_step: {self.inference_step}" , "magenta")
        cprint(f"denoise_time_all: {self.denoise_time_all}" , "magenta")
        cprint(f"bootstrap_every: {self.bootstrap_every}" , "magenta")
        cprint(f"use_shortcut: {self.use_shortcut}" , "magenta")
        cprint(f"atol: {atol}" , "magenta")
        cprint(f"self.inference_step: {self.inference_step}" , "magenta")
        
    # ========= inference  ============

    '''
    if we use the original setting as dp3, then 

    condition_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    condition_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    然后 local_cond = None, 只有global_cond

    '''


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        nobs = self.normalizer.normalize(obs_dict, exclude_keys=['contact_forces'])
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        if self.use_contact_forces: 
            # convert the continuous forces to binary contact, separated by 0.1

            if nobs['contact_forces'].shape[-1] == 3:
                nobs['contact_forces'] = torch.norm(nobs['contact_forces'], p=2, dim=-1, keepdim=True).squeeze(-1)
            nobs['contact_forces'] = (nobs['contact_forces'] > 0.1).float()
        this_n_point_cloud = nobs['point_cloud']
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        

        # run sampling, generation
        # 0. sample noise
        trajectory = torch.randn(
        size=cond_data.shape, 
        dtype=cond_data.dtype,
        device=cond_data.device)
        # 1. apply conditioning
        trajectory[cond_mask] = cond_data[cond_mask]

        # 2. predict t and move model forward
        denoise_timesteps = torch.tensor(self.inference_step, dtype=torch.int32, device=self.device)
        delta_t = 1.0 / denoise_timesteps
        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps
            t_vector = torch.full((B,), t, device=device, dtype=dtype)
            if not self.use_shortcut:
                dt_flow = torch.log2(torch.tensor(self.denoise_time_all, dtype=torch.int32, device=self.device)).to(dtype=torch.int32, device=self.device)
                dt_base = torch.ones(B, dtype=torch.int32, device=self.device) * dt_flow  # smallest dt
            else:
                dt_flow = torch.log2(denoise_timesteps).to(dtype=torch.int32, device=self.device)
                dt_base = torch.ones(B, dtype=torch.int32, device=self.device) * dt_flow  
            
            with torch.no_grad():
                model_output = self.model.forward(sample=trajectory,timestep=t_vector, dt=dt_base,
                                                  local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute next trajectory
            trajectory = trajectory + delta_t * model_output # Euler step
        

        nsample = trajectory

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction

        # change for now, calculate the avg inference time
        # global cnt_time_list
        result = {
            'action': action,
            'action_pred': action_pred,
            # 'avg_inference_time': torch.tensor(cnt_time_list, device=self.device)
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_shortcut_loss(self, batch):
        # normalize input
        # nobs = self.normalizer.normalize(batch['obs'])
        nobs = self.normalizer.normalize(batch['obs'], exclude_keys=['contact_forces'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

                    # update the binary contact forces into nobs if we use contact forces
        if self.use_contact_forces: 

            if nobs['contact_forces'].shape[-1] == 3:
                nobs['contact_forces'] = torch.norm(nobs['contact_forces'], p=2, dim=-1, keepdim=True).squeeze(-1)
                # nobs['contact_forces'] = nobs['contact_forces']
            nobs['contact_forces'] = (nobs['contact_forces'] > 0.1).float()



        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:  # obs_as_global_cond: True
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        # compute the loss of the flow matching model
        x0 = noise  # in the original file it writes: x0 = torch.rand_like(x1)
        x1 = trajectory
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        x0[condition_mask] = cond_data[condition_mask]

        '''
        torch.all(loss_mask): True   -> all true
        torch.any(condition_mask): False  -> all false
        '''

        # sample timesteps through the flow matcher                
        pred_type = self.prediction_type 
        assert(pred_type == 'conditional_flow_matching')

        # 1. sample dt
        denoise_timesteps = torch.tensor(self.denoise_time_all, dtype=torch.int32, device=self.device)
        bootstrap_batchsize = torch.tensor(batch_size // self.bootstrap_every, dtype=torch.int32, device=self.device)
        log2_sections = torch.log2(denoise_timesteps).to(self.device)

        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections).to(self.device), (bootstrap_batchsize // log2_sections).int())
        # 因为这里有可能在那个整除的环节，使得dt_base和bst_batchsz并不相等，所以剩下的我们用0补齐
        dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batchsize - dt_base.shape[0], dtype=torch.float32, device=self.device)])
        num_dt_cfg = bootstrap_batchsize // log2_sections

        # no force_dts
        dt = 1 / (2 ** (dt_base)) # should be [1, 1/2, 1/4, 1/8, 1/16, 1/32]
        dt_base_bootstrap = dt_base + 1
    
        dt_bootstrap = dt / 2

        # 2. sample t
        dt_sections = torch.pow(2, dt_base).to(dtype=torch.int32) # [1, 2, 4, 8, 16, 32]
        
        # t = torch.randint(torch.zeros_like(dt_sections), dt_sections.int(), (bootstrap_batchsize,)).float()
        t = torch.tensor([torch.randint(0, dt_sections[i].item(), (1,)) for i in range(bootstrap_batchsize)], dtype=self.dtype, device=self.device)
        t = t / dt_sections # Between 0 and 1.
        # no force t here
        # t_full = t[:, None, None, None]


        t_full = t[:, None, None]    # modified here, I dont know why it is 4D?  Even to generate images?

        # 3. generate bootstrap targets

        x_1 = x1[:bootstrap_batchsize]
        x_0 = x0[:bootstrap_batchsize]
        bst_local_cond = local_cond[:bootstrap_batchsize] if local_cond is not None else None
        bst_global_cond = global_cond[:bootstrap_batchsize] if global_cond is not None else None


        assert(x_0.shape == x_1.shape)
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        # for i in range(t.shape[0]):
        #     # cprint(f"{x_t[i, ...] }", "grey")        
        #     # cprint(f"{x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...])}", "grey")
        #     if not torch.isclose(x_t[i, ...], x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...]), rtol=0.01).all():
        #         false_count = torch.sum(~torch.isclose(x_t[i, ...], x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...]), rtol=0.01)).item()
        '''
        xt.shape: torch.Size([16, 4, 28])
        bst_global_cond.shape: torch.Size([16, 256])
        对的对的，因为你看2个，每个的dim是128,reshape就是256了
        '''


        # not bootstrap cfg
        self.model.eval()
        with torch.no_grad():  # stop grad
            v_b1 = self.model(
                sample=x_t, 
                timestep=t, 
                dt = dt_base_bootstrap,
                local_cond=bst_local_cond, 
                global_cond=bst_global_cond)
        self.model.train()


        t2 = t + dt_bootstrap
        # x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = x_t + dt_bootstrap[:, None, None] * v_b1  # modified here
        x_t2 = torch.clip(x_t2, -4, 4)

        self.model.eval()
        with torch.no_grad():  # stop grad 应该是没有问题的，因为你本身v_target对标的就是v_t = x1 - x0，所以本身也是一个没有梯度的标量咯
            v_b2 = self.model(
                sample=x_t2, 
                timestep=t2, 
                dt = dt_base_bootstrap,
                local_cond=bst_local_cond, 
                global_cond=bst_global_cond)
        self.model.train()

        v_target = (v_b1 + v_b2) / 2
        
        v_target = torch.clip(v_target, -4, 4)
        bst_v = copy.deepcopy(v_target)
        bst_dt = copy.deepcopy(dt_base)
        bst_t = copy.deepcopy(t)
        bst_xt = copy.deepcopy(x_t)

        # 4. generate flow matching targets
        # no class dropout prob here
        # sample t
        t = torch.randint(0, denoise_timesteps, (x1.shape[0],), device=self.device).float()
        t /= denoise_timesteps
        # no force t here
        # t_full = t[:, None, None, None]
        t_full = t[:, None, None] # explain is in note.md

        # sample flow pairs x_t, v_t
        x_0 = torch.randn(x1.shape, device=x1.device)
        x_1 = x1
        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
        # for i in range(t.shape[0]):
        #     # cprint(f"{x_t[i, ...] }", "blue")        
        #     # cprint(f"{x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...])}", "blue")
        #     if not torch.isclose(x_t[i, ...], x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...]), rtol=0.01).all():
        #         false_count = torch.sum(~torch.isclose(x_t[i, ...], x_0[i, ...] + t[i] * (x_1[i, ...] - x_0[i, ...]), rtol=0.01)).item()

        v_t = x_1 - (1 - 1e-5) * x_0
        dt_flow = torch.log2(denoise_timesteps).to(dtype=torch.int32, device=self.device)
        dt_base = torch.ones(x1.shape[0], dtype=torch.int32, device=self.device) * dt_flow

        # v_origin_prime = self.model(
        #     sample = x_t,
        #     timestep = t,
        #     dt = dt_base,
        #     local_cond=local_cond,
        #     global_cond=global_cond,       
        # )
        # loss_origin_prime = F.mse_loss(v_origin_prime, v_t, reduction='none')
        # loss_origin_prime = loss_origin_prime * loss_mask.type(loss_origin_prime.dtype)
        # loss_origin_prime = reduce(loss_origin_prime, 'b ... -> b (...)', 'mean')
        # loss_origin_prime = loss_origin_prime.mean()

        # 5. merge flow+bootstrap
        bst_size = batch_size // self.bootstrap_every
        # bst_size_data = batch_size - bst_size

        
        assert(bst_size == bootstrap_batchsize.item())

        x_t_all = torch.cat([bst_xt, x_t[bst_size:]], dim=0)
        t_all = torch.cat([bst_t, t[bst_size:]], dim=0)
        dt_base_all = torch.cat([bst_dt, dt_base[bst_size:]], dim=0)
        v_t_all = torch.cat([bst_v, v_t[bst_size:]], dim=0)
        local_cond_all = torch.cat([bst_local_cond, local_cond[bst_size:]], dim=0) if local_cond is not None else None
        global_cond_all = torch.cat([bst_global_cond, global_cond[bst_size:]], dim=0) if global_cond is not None else None

        

        # v_part_flow_prime = self.model(
        #     sample = x_t[bst_size:],
        #     timestep = t[bst_size:],
        #     dt = dt_base[bst_size:],
        #     local_cond=local_cond[bst_size:] if local_cond is not None else None,
        #     global_cond=global_cond[bst_size:] if global_cond is not None else None,       
        # )
        # loss_part_flow_origin_prime = F.mse_loss(v_part_flow_prime, v_t[bst_size:], reduction='none')
        # loss_part_flow_origin_prime = loss_part_flow_origin_prime * loss_mask[bst_size:].type(loss_part_flow_origin_prime.dtype)
        # loss_part_flow_origin_prime = reduce(loss_part_flow_origin_prime, 'b ... -> b (...)', 'mean')
        # loss_part_flow_origin_prime = loss_part_flow_origin_prime.mean()


        # v_all_prime = self.model(
        #     sample = x_t_all[bst_size:],
        #     timestep = t_all[bst_size:],
        #     dt = dt_base_all[bst_size:],
        #     local_cond=local_cond[bst_size:] if local_cond is not None else None,
        #     global_cond=global_cond[bst_size:] if global_cond is not None else None,       
        # )
        # loss_all_flow_prime = F.mse_loss(v_all_prime, v_t_all[bst_size:], reduction='none')
        # loss_all_flow_prime = loss_all_flow_prime * loss_mask[bst_size:].type(loss_all_flow_prime.dtype)
        # loss_all_flow_prime = reduce(loss_all_flow_prime, 'b ... -> b (...)', 'mean')
        # loss_all_flow_prime = loss_all_flow_prime.mean()

        # no labels dropped
        

        info = {}
        info['bootstrap_ratio'] = torch.mean((dt_base != dt_flow).float())

        info['v_magnitude_bootstrap'] = torch.sqrt(torch.mean(bst_v.pow(2)))
        info['v_magnitude_b1'] = torch.sqrt(torch.mean(v_b1.pow(2)))
        info['v_magnitude_b2'] = torch.sqrt(torch.mean(v_b2.pow(2)))

        # return x_t, v_t, t, dt_base, info

        v_prime = self.model(
            sample = x_t_all,
            timestep = t_all,
            dt = dt_base_all,
            local_cond=local_cond_all,
            global_cond=global_cond_all,       
        )
        loss = F.mse_loss(v_prime, v_t_all, reduction='none')
        # means that the reduction will not be taken in mse_loss step, but is taken in the einop.reduce later    
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_boot = F.mse_loss(v_prime[:bst_size], v_t_all[:bst_size], reduction='none')
        loss_boot = loss_boot * loss_mask[:bst_size].type(loss.dtype)
        loss_boot = reduce(loss_boot, 'b ... -> b (...)', 'mean')
        loss_boot = loss_boot.mean()

        loss_flow = F.mse_loss(v_prime[bst_size:], v_t_all[bst_size:], reduction='none')
        loss_flow = loss_flow * loss_mask[bst_size:].type(loss.dtype)
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean')
        loss_flow = loss_flow.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
                "loss_boot": loss_boot,
                "loss_flow": loss_flow,
                'v_magnitude_prime': torch.sqrt(torch.mean(v_prime.pow(2))),
            }
        
        return loss, loss_dict





