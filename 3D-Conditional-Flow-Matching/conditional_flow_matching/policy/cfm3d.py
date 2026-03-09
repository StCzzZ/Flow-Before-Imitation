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
from conditional_flow_matching.model.conditional_unet.conditional_unet1d import ConditionalUnet1D
from conditional_flow_matching.model.conditional_unet.mask_generator import LowdimMaskGenerator
from conditional_flow_matching.common.pytorch_util import dict_apply, dict_apply_print
from conditional_flow_matching.common.model_util import print_params
from conditional_flow_matching.model.vision.pointnet_extractor import CFMEncoder, PCTactileEncoder, PointNetTactileEncoder, PointNetTactileLargeEncoder



# addition

import torchdiffeq

# global check variant 

# cnt_time = 0
# cnt_time_list = []



class CFM3D(BasePolicy):
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
                                    out_channel=encoder_output_dim,   # set = 64 in cfg
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
        elif encoder_type == "Fused":
            obs_encoder = PCTactileEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                use_contact_forces=use_contact_forces,
            )
        elif encoder_type == "Pointnet-Fused":
            obs_encoder = PointNetTactileEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                use_contact_forces=use_contact_forces,
            )
        elif encoder_type == "Pointnet-Large":
            obs_encoder = PointNetTactileLargeEncoder(
                observation_space=obs_dict,
                out_channel=encoder_output_dim,
                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                use_pc_color=use_pc_color,
                pointnet_type=pointnet_type,
                use_contact_forces=use_contact_forces,
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
        cprint(f"[CFM3D] flow_matching_embed_dim: {flow_matching_embed_dim}", "yellow")



        model = ConditionalUnet1D(
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
        self.atol = atol
        self.rtol = rtol

        # eps:  the The smallest time step to sample from, imitating https://github.com/gnobitab/RectifiedFlow
        # we continue to use this setting, avoid the exact "0" here
        self.T = 1.
        self.eps = 1e-4


        self.kwargs = kwargs



        # we don't need num_inference steps in flow matching
        # if num_inference_steps is None:
        #     num_inference_steps = noise_scheduler.config.num_train_timesteps
        # self.num_inference_steps = num_inference_steps


        print_params(self)
        cprint(f"Model_name: {self.__class__.__name__}", "magenta")
        cprint(f"flow_matcher: {flow_matcher}" , "magenta")
        cprint(f"prediction_type: {prediction_type}" , "magenta")
        cprint(f"inference_step: {inference_step}" , "magenta")
        cprint(f"atol: {atol}" , "magenta")
        cprint(f"self.inference_step: {self.inference_step}" , "magenta")
        
    # ========= inference  ============

    '''
    if we use the original setting as dp3, then 

    condition_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
    condition_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    然后 local_cond = None, 只有global_cond

    '''

    # we don't need to change the whole function here, we only have to keep the correspondence of "inference_step" and the "target_step"

    def conditional_flow_matching(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None, inference_step=5, 
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        # global cnt_time
        

        # adapted from https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py
        def forward_model(t, trajectory): # it's the modification of the original lambda function, as it's a little bit hard to be implemented in a single lambda func
            
            # global cnt_time
            # # cprint(f"trajectory[condition_mask]: {trajectory[condition_mask]}", "blue")
            # # !!!! 这里timestep变成"数字了"，所以shape是[]
            # # cprint(f"t.shape: {t.shape}", "yellow")

            '''
            torch.any(condition_mask): False
            condition_mask.shape: torch.Size([1, 4, 26])
            trajectory.shape: torch.Size([])
            '''

            # # 0. return to the original dimension
            # trajectory = trajectory.squeeze(1)

            t = torch.tensor([t], device=self.device, dtype=self.dtype)

            # cnt_time += 1

            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            # 2. forward model
            model_output = model.forward(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            # 3. compute next trajectory
            trajectory = model_output
            # 4. make sure conditioning is enforced
            trajectory[condition_mask] = condition_data[condition_mask]   

            return trajectory
        # '''
        # torch.Size([1, 2, 26]  在原来的地方是4D 的，定义是 B, C, H, W, 感觉是因为这个问题所以t搞不出来
        # 嗯.. 不是这个问题
        # '''
        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb
        # traj = torchdiffeq.odeint(
        #     func = forward_model,
        #     y0 = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,device=condition_data.device),   # 给它加一个channel放进去
        #     t = torch.linspace(0, 1, 2, device=condition_data.device),
        #     atol=1e-4,
        #     rtol=1e-4,
        #     method="dopri5",
        # )
        '''
        t_array: tensor([1.0000e-04, 1.1120e-01, 2.2230e-01, 3.3340e-01, 4.4450e-01, 5.5560e-01,
        6.6670e-01, 7.7780e-01, 8.8890e-01, 1.0000e+00], device='cuda:0')
        '''

        # t = 1e-4 ~ 0.0889, that's common, if we reduce the infer_step to 1, then we only need to predict the velocity at time 0
        # we don't need to predict the velocity at time 1
        # !!! changed to generalize the cfg

        # here +1 is to make the inference step consistent. if infer_step = 1, then linspace(0, 1, 1) will fail. 
        # so we're using (0, 1, 2) here
        with torch.no_grad():
            traj = torchdiffeq.odeint(
                func = forward_model,
                y0 = torch.randn(size=condition_data.shape, dtype=condition_data.dtype,device=condition_data.device),   # 给它加一个channel放进去
                t = torch.linspace(0, 1, inference_step+1, device=condition_data.device) * (self.T - self.eps) + self.eps,
                atol=self.atol,
                rtol=self.rtol,
                method=self.ode_method,
            )
        # cnt_time_list.append(cnt_time)
        # cnt_time = 0

        # the output of the "odeint" module is k-Dimentional, where k depends on the timesteps that we choose for the param t
        # for exp, if we choose t = torch.linspace(0, 1, 2, device=condition_data.device), then the output will be 2D, representing the 
        # trajectory at time 0 and 1 respectively

        traj = traj[-1]  # get the last timestep

        return traj


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict, exclude_keys=['contact_forces'])
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        if self.use_contact_forces: 
            # convert the continuous forces to binary contact, separated by 0.1
            if nobs['contact_forces'].shape[-1] == 3:
                nobs['contact_forces'] = torch.norm(nobs['contact_forces'], p=2, dim=-1, keepdim=True).squeeze(-1)
                # nobs['contact_forces'] = nobs['contact_forces']
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
        if self.obs_as_global_cond:  # True
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:  # film feature-wise ...
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

        # run sampling
        nsample = self.conditional_flow_matching(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            inference_step=self.inference_step, 
            **self.kwargs)


        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        # result = {
        #     'action': action,
        #     'action_pred': action_pred,
        # }

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

    #TODO: change the loss function 
    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'], exclude_keys=['contact_forces'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        # update the binary contact forces into nobs if we use contact forces
        if self.use_contact_forces: 
            # convert the continuous forces to binary contact, separated by 0.1
            # pre_process the "contact_force", if it is 3-direction, turn it into norm form, and into binary: 
            # if nobs['contact_forces'].shape[-1] == 3:
            #     nobs['contact_forces'] = torch.norm(nobs['contact_forces'], dim=-1, keepdim=True)
            # assert (nobs['contact_forces'].shape[-1] == 1)
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
            # 
            # # self.use_pc_color: False, this_nobs['point_cloud'].shape: torch.Size([174, 512, 3])
            # self.use_pc_color: True, this_nobs['point_cloud'].shape: torch.Size([256, 1024, 6]), this_n_obs['point_cloud].range: -1.0 ~ 1.0

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

        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb

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
        if pred_type == 'conditional_flow_matching':
            
            # https://github.com/atong01/conditional-flow-matching
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
            # predict vt
            vt = self.model(
                sample=xt, 
                timestep=t, 
                local_cond=local_cond, 
                global_cond=global_cond)
            target = ut
            pred = vt
        
        elif pred_type == 'optimal_transport_conditional_flow_matching':
            # https://github.com/atong01/conditional-flow-matching
            t, xt, ut, idx, jdx = self.flow_matcher.classifier_free_sample_location_and_conditional_flow(x0, x1)

            # apply the index to the conditions to get the corresponding conditions of xt

            # the condition should be correspond to x1(i.e. the gt-traj), for x0 is merely a noise, wherever the "noise"
            # is taken, it doesn't matter, but we have to make sure that the x_t with the corresponding condition would
            # reach the correct "traj".                                                                        
                           
            if local_cond is not None:
                local_cond_t = local_cond[jdx]
            else:
                local_cond_t = None

            if global_cond is not None: 
                global_cond_t = global_cond[jdx]
            else: 
                global_cond_t = None
            
            '''
            local_cond is None
            global_cond.shape: torch.Size([13, 256])
            global_cond_t.shape: torch.Size([13, 256])

            the "idx, jdx" term is only for selecting the corresponding x0 and x1s

            and the "condition" is with respect to the currtent step xt so we should use idx to select the condition
            '''

            # predict vt
            vt = self.model(
                sample=xt, 
                timestep=t, 
                local_cond=local_cond_t, 
                global_cond=global_cond_t)
            
            target = ut
            pred = vt

        

        loss = F.mse_loss(pred, target, reduction='none')
        # means that the reduction will not be taken in mse_loss step, but is taken in the einop.reduce later    
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict
    

        #TODO: change the loss function 
    #Done
    def compute_reflow_loss(self, batch, batch_z1):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
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

        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb

        # compute the loss of the flow matching model
        x0 = noise  # in the original file it writes: x0 = torch.rand_like(x1)
        x1 = trajectory
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        x0[condition_mask] = cond_data[condition_mask]
        pred_type = self.prediction_type 
        #TODO: change the loss function, i.e., u_t  -> z_1
        '''
            loss term: 

            |(Z_1 - Z_0) - v_\theta(Z_t, t)|^2
        '''
        if pred_type == 'conditional_flow_matching':
            
            # get z1
            z1 = batch_z1

            # https://github.com/gnobitab/RectifiedFlow
            # loss_func = mse((z1 - z0) - v_t(z_t, t)), Z_t = t Z_1 + (1-t) Z_0, where t is arbitrary, rand_like

            # https://github.com/atong01/conditional-flow-matching
            t, zt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, z1)
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond, 
                global_cond=global_cond)

            target = z1 - x0   # actually, if we look into the "sample_location" function, we'll find out z1 - x0 == ut
            # # here I just wrote z1 - x0 in order to make the code more clear. 

            # change to check the gradient backward
            # target = torch.zeros_like(z1-x0, device=self.device, dtype=self.dtype)
            pred = vt
        
        elif pred_type == 'optimal_transport_conditional_flow_matching':

            # get z1
            z1 = batch_z1

            
            # https://github.com/atong01/conditional-flow-matching
            t, zt, ut, idx, jdx = self.flow_matcher.classifier_free_sample_location_and_conditional_flow(x0, z1)
                                                                      
            if local_cond is not None:
                local_cond_t = local_cond[jdx]
            else:
                local_cond_t = None
            if global_cond is not None: 
                global_cond_t = global_cond[jdx]
            else: 
                global_cond_t = None
            
            # make z1 with the same idx with vt
            z1 = z1[jdx]
            
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond_t, 
                global_cond=global_cond_t)

            target = z1 - x0  # the same as above
            pred = vt
        with torch.no_grad():
            loss_2 = F.mse_loss(z1, x1, reduction='none')
            loss_2 = loss_2 * loss_mask.type(loss_2.dtype)
            loss_2 = reduce(loss_2, 'b ... -> b (...)', 'mean')
            loss_2 = loss_2.mean()
        # with torch.no_grad():
        #     loss_3 = F.mse_loss(target, ut, reduction='none')   # check whether target == ut
        #     loss_3 = loss_3 * loss_mask.type(loss_3.dtype)
        #     loss_3 = reduce(loss_3, 'b ... -> b (...)', 'mean')
        #     loss_3 = loss_3.mean()
        #     (f"loss between target and ut: {loss_3.item()}", "magenta")    # 0.0, they are the same

        with torch.no_grad():
            loss_4 = F.mse_loss(vt, x1-x0, reduction='none')
            loss_4 = loss_4 * loss_mask.type(loss_4.dtype)
            loss_4 = reduce(loss_4, 'b ... -> b (...)', 'mean')
            loss_4 = loss_4.mean()

        '''
        z1.shape: torch.Size([128, 4, 26])
        x1.shape: torch.Size([128, 4, 26])
        loss_2: 0.0006109100650064647   对啊，这个z1 也就和 x1 差距这么小，为什么vt 的loss会越训越大呢...
        把这个挂上去试试看

        true model, don't worry
        '''

        loss = F.mse_loss(pred, target, reduction='none')
        # means that the reduction will not be taken in mse_loss step, but is taken in the einop.reduce later    
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {
                'bc_loss': loss.item(),
                'loss_z1-x1': loss_2.item(),
                'loss_vt_x1-x0': loss_4.item(),
            }
        
        return loss, loss_dict


    def compute_distill_loss(self, batch, batch_z1):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
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

        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb

        # compute the loss of the flow matching model
        x0 = noise  # in the original file it writes: x0 = torch.rand_like(x1)
        x1 = trajectory
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        x0[condition_mask] = cond_data[condition_mask]
        pred_type = self.prediction_type 
        #TODO: change the loss function, i.e., u_t  -> z_1
        '''
            loss term: 

            |(Z_1 - Z_0) - v_\theta(Z_t, t)|^2
        '''
        if pred_type == 'conditional_flow_matching':
            
            # get z1
            z1 = batch_z1

            # https://github.com/gnobitab/RectifiedFlow
            # loss_func = mse((z1 - z0) - v_t(z_t, t)), Z_t = t Z_1 + (1-t) Z_0, where t satisfies： 
            # t = torch.randint(0, sde.reflow_t_schedule, (batch.shape[0], ), device=batch.device) * (sde.T - eps) / sde.reflow_t_schedule + eps

            # only one-line difference from the "reflow_loss". 
            t = torch.randint(0, self.inference_step, size=(batch_size, ), device=trajectory.device) * (self.T - self.eps) + self.eps

            _, zt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, z1, t=t)

            
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond, 
                global_cond=global_cond)

            target = z1 - x0   # actually, if we look into the "sample_location" function, we'll find out z1 - x0 == ut
            # # here I just wrote z1 - x0 in order to make the code more clear. 

            # change to check the gradient backward
            # target = torch.zeros_like(z1-x0, device=self.device, dtype=self.dtype)
            pred = vt
        
        elif pred_type == 'optimal_transport_conditional_flow_matching':

            # get z1
            z1 = batch_z1

            
            # https://github.com/atong01/conditional-flow-matching

            t = torch.randint(0, self.inference_step, size=(batch_size, ), device=trajectory.device) * (self.T - self.eps) + self.eps

            _, zt, ut, idx, jdx = self.flow_matcher.classifier_free_sample_location_and_conditional_flow(x0, z1, t=t)
                                                                      
            if local_cond is not None:
                local_cond_t = local_cond[jdx]
            else:
                local_cond_t = None
            if global_cond is not None: 
                global_cond_t = global_cond[jdx]
            else: 
                global_cond_t = None
            
            # make z1 with the same idx with vt
            z1 = z1[jdx]
            
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond_t, 
                global_cond=global_cond_t)

            target = z1 - x0  # the same as above
            pred = vt
        with torch.no_grad():
            loss_2 = F.mse_loss(z1, x1, reduction='none')
            loss_2 = loss_2 * loss_mask.type(loss_2.dtype)
            loss_2 = reduce(loss_2, 'b ... -> b (...)', 'mean')
            loss_2 = loss_2.mean()
        # with torch.no_grad():
        #     loss_3 = F.mse_loss(target, ut, reduction='none')   # check whether target == ut
        #     loss_3 = loss_3 * loss_mask.type(loss_3.dtype)
        #     loss_3 = reduce(loss_3, 'b ... -> b (...)', 'mean')
        #     loss_3 = loss_3.mean()

        with torch.no_grad():
            loss_4 = F.mse_loss(vt, x1-x0, reduction='none')
            loss_4 = loss_4 * loss_mask.type(loss_4.dtype)
            loss_4 = reduce(loss_4, 'b ... -> b (...)', 'mean')
            loss_4 = loss_4.mean()
            cprint(f"loss_4 between vt and x1-x0: {loss_4.item()}", "magenta")  

        '''
        z1.shape: torch.Size([128, 4, 26])
        x1.shape: torch.Size([128, 4, 26])
        loss_2: 0.0006109100650064647   对啊，这个z1 也就和 x1 差距这么小，为什么vt 的loss会越训越大呢...
        把这个挂上去试试看

        true model, don't worry
        '''

        loss = F.mse_loss(pred, target, reduction='none')
        # means that the reduction will not be taken in mse_loss step, but is taken in the einop.reduce later    
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_dict = {
                'bc_loss': loss.item(),
                'loss_z1-x1': loss_2.item(),
                'loss_vt_x1-x0': loss_4.item(),
            }
        
        return loss, loss_dict
    
    #TODO: change the loss function 
    #Done
    def compute_reflow_loss_backup(self, batch, teacher_model: BasePolicy):
        # normalize input

        teacher_model.eval()  # the teacher model should be in eval mode

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
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

        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb

        # compute the loss of the flow matching model
        x0 = noise  # in the original file it writes: x0 = torch.rand_like(x1)
        x1 = trajectory
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        x0[condition_mask] = cond_data[condition_mask]
        pred_type = self.prediction_type 
        #TODO: change the loss function, i.e., u_t  -> z_1
        '''
            loss term: 

            |(Z_1 - Z_0) - v_\theta(Z_t, t)|^2
        '''
        if pred_type == 'conditional_flow_matching':
            
            # get z1
            with torch.no_grad():
                z1 = teacher_model.conditional_flow_matching(
                    cond_data, 
                    condition_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    inference_step=self.inference_step,
                    **self.kwargs)

            # https://github.com/gnobitab/RectifiedFlow
            # loss_func = mse((z1 - z0) - v_t(z_t, t)), Z_t = t Z_1 + (1-t) Z_0, where t is arbitrary, rand_like

            # https://github.com/atong01/conditional-flow-matching
            t, zt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, z1)
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond, 
                global_cond=global_cond)
            
            # add!!!
            z1 = z1.detach()

            target = z1 - x0   # actually, if we look into the "sample_location" function, we'll find out z1 - x0 == ut
            # # here I just wrote z1 - x0 in order to make the code more clear. 
            # change to check the gradient backward
            # target = torch.zeros_like(z1-x0, device=self.device, dtype=self.dtype)
            pred = vt
        
        elif pred_type == 'optimal_transport_conditional_flow_matching':

            # get z1
            with torch.no_grad():
                z1 = teacher_model.conditional_flow_matching(
                    cond_data, 
                    condition_mask,
                    local_cond=local_cond,
                    global_cond=local_cond,
                    inference_step=self.inference_step,
                    **self.kwargs)

            
            # https://github.com/atong01/conditional-flow-matching
            t, zt, ut, idx, jdx = self.flow_matcher.classifier_free_sample_location_and_conditional_flow(x0, z1)
                                                                      
            if local_cond is not None:
                local_cond_t = local_cond[jdx]
            else:
                local_cond_t = None
            if global_cond is not None: 
                global_cond_t = global_cond[jdx]
            else: 
                global_cond_t = None
            
            # make z1 with the same idx with vt
            z1 = z1[jdx]
            
            # predict vt
            vt = self.model(
                sample=zt, 
                timestep=t, 
                local_cond=local_cond_t, 
                global_cond=global_cond_t)

            
            
            target = z1 - x0  # the same as above
            pred = vt
        with torch.no_grad():
            loss_2 = F.mse_loss(z1, x1, reduction='none')
            loss_2 = loss_2 * loss_mask.type(loss_2.dtype)
            loss_2 = reduce(loss_2, 'b ... -> b (...)', 'mean')
            loss_2 = loss_2.mean()
            cprint(f"loss between z1 and x1: {loss_2.item()}", "yellow")
        '''
        z1.shape: torch.Size([128, 4, 26])
        x1.shape: torch.Size([128, 4, 26])
        loss_2: 0.0006109100650064647   对啊，这个z1 也就和 x1 差距这么小，为什么vt 的loss会越训越大呢...
        把这个挂上去试试看

        '''

        loss = F.mse_loss(pred, target, reduction='none')
        # means that the reduction will not be taken in mse_loss step, but is taken in the einop.reduce later    
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict
    
    def generate_z1(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        # def f(key, value):
        # dict_apply_print(nobs, f)
            # return value

        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
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

        # adapted from https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/conditional_mnist.ipynb

        # compute the loss of the flow matching model
        x0 = noise  # in the original file it writes: x0 = torch.rand_like(x1)
        x1 = trajectory
        # compute loss mask
        loss_mask = ~condition_mask
        # apply conditioning
        x0[condition_mask] = cond_data[condition_mask]
        pred_type = self.prediction_type 

        # get z_1
        if pred_type == 'conditional_flow_matching':
            # get z1
            with torch.no_grad():
                z1 = self.conditional_flow_matching(
                    cond_data, 
                    condition_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    inference_step=self.inference_step,
                    **self.kwargs)
        
        elif pred_type == 'optimal_transport_conditional_flow_matching':
            # get z1
            with torch.no_grad():
                z1 = self.conditional_flow_matching(
                    cond_data, 
                    condition_mask,
                    local_cond=local_cond,
                    global_cond=local_cond,
                    inference_step=self.inference_step,
                    **self.kwargs)
            # https://github.com/atong01/conditional-flow-matching
            t, zt, ut, idx, jdx = self.flow_matcher.classifier_free_sample_location_and_conditional_flow(x0, z1)                                               
            if local_cond is not None:
                local_cond_t = local_cond[jdx]
            else:
                local_cond_t = None
            if global_cond is not None: 
                global_cond_t = global_cond[jdx]
            else: 
                global_cond_t = None
            # make z1 with the same idx with vt
            z1 = z1[jdx]
            

        else: 
            raise NotImplementedError(f"pred_type: {pred_type} is not implemented yet")

        # with torch.no_grad():
        #     loss_2 = F.mse_loss(z1, x1, reduction='none')
        #     loss_2 = loss_2 * loss_mask.type(loss_2.dtype)
        #     loss_2 = reduce(loss_2, 'b ... -> b (...)', 'mean')
        #     loss_2 = loss_2.mean()
        
        return z1

