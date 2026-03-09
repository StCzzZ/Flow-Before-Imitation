import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

from .pointnet import PointNetfeat


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')

        # norm layer won't change the shape, only the value
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]  # 对x的第一个维度做maxpooling了，1024 -> 1. 
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),   # use layer norm true here
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class CFMEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 use_contact_forces = False, 
                 encoder_type = "CFM3D", # can be CFM3D, Fused or Pointnet-Fused or Pointnet-Large
                 contact_mlp_size=(64, 64), contact_mlp_activation_fn=nn.ReLU,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'

        self.contact_forces_key = 'contact_forces'

        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if use_contact_forces:
            self.contact_shape = observation_space[self.contact_forces_key]

        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[CFMEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CFMEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[CFMEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        cprint(f"[PC ENCODER CFG]pointcloud_encoder_cfg: {pointcloud_encoder_cfg}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.use_contact_forces = use_contact_forces
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim  # set to 64 in defalt params
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        if use_contact_forces:
            if len(contact_mlp_size) == 0:
                raise RuntimeError(f"Contact mlp size is empty")
            elif len(contact_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = contact_mlp_size[:-1]
            output_dim = contact_mlp_size[-1]

            self.n_output_channels  += output_dim  # set to 64 in defalt params
            self.contact_mlp = nn.Sequential(*create_mlp(self.contact_shape[0], output_dim, net_arch, contact_mlp_activation_fn))


        cprint(f"[CFMEncoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[CFMEncoder] use_contact_forces: {use_contact_forces}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:  # false in isaac cube case
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))

        '''
        [CFMEncoder] points shape: torch.Size([2, 1024, 6])
        [CFMEncoder] pn_feat shape: torch.Size([2, 64])
        [CFMEncoder] state.shape: torch.Size([2, 24])
        [CFMEncoder] state_feat shape: torch.Size([2, 64])
        '''
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)

        if self.use_contact_forces:
            contact = observations[self.contact_forces_key]
            contact_feat = self.contact_mlp(contact)
            final_feat = torch.cat([final_feat, contact_feat], dim=-1)

            '''
            [CFMEncoder] contact_feat shape: torch.Size([256, 64])
            [CFMEncoder] final_feat shape: torch.Size([256, 192])
            '''

        return final_feat


    def output_shape(self):
        return self.n_output_channels

class PCTactileEncoder(nn.Module):
    def __init__(self, 
        observation_space: Dict, 
        # img_crop_shape=None,
        out_channel=256,
        state_force_mlp_size=(64, 64), state_force_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type='pointnet',
        use_contact_forces = False, 
        # contact_mlp_size=(64, 64), contact_mlp_activation_fn=nn.ReLU,
        ):
        super().__init__()
        # self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        # self.rgb_image_key = 'image'

        self.contact_forces_key = 'contact_forces'

        self.n_output_channels = out_channel
        
        # self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if use_contact_forces:
            self.contact_shape = observation_space[self.contact_forces_key]
            self.state_contact_shape = self.state_shape[0] + self.contact_shape[0]
        
        else: 
            self.state_contact_shape = self.state_shape[0]

        # if self.use_imagined_robot:
        #     self.imagination_shape = observation_space[self.imagination_key]
        # else:
        self.imagination_shape = None
            
    
        cprint(f"[CFMEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CFMEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[CFMEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        cprint(f"[PC ENCODER CFG]pointcloud_encoder_cfg: {pointcloud_encoder_cfg}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.use_contact_forces = use_contact_forces
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        # if len(state_mlp_size) == 0:
        #     raise RuntimeError(f"State mlp size is empty")
        # elif len(state_mlp_size) == 1:
        #     net_arch = []
        assert len(state_force_mlp_size) >= 2
        # else:

        # concat joint_state and contact_mlp together
        net_arch = state_force_mlp_size[:-1]
        output_dim = state_force_mlp_size[-1]

        self.n_output_channels  += output_dim  # set to 64 in defalt params
        self.state_force_mlp = nn.Sequential(*create_mlp(self.state_contact_shape, output_dim, net_arch, state_force_mlp_activation_fn))

        cprint(f"[CFMEncoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[CFMEncoder] use_contact_forces: {use_contact_forces}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:  # false in isaac cube case
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))

        '''
        [CFMEncoder] points shape: torch.Size([2, 1024, 6])
        [CFMEncoder] pn_feat shape: torch.Size([2, 64])
        [CFMEncoder] state.shape: torch.Size([2, 24])
        [CFMEncoder] state_feat shape: torch.Size([2, 64])
        '''
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        contact = observations[self.contact_forces_key]

        state_force = torch.cat([state, contact], dim=-1)
        state_force_feat = self.state_force_mlp(state_force)  # B * 64
        final_feat = torch.cat([pn_feat, state_force_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
    

class PointNetTactileEncoder(nn.Module):
    def __init__(self, 
        observation_space: Dict, 
        # img_crop_shape=None,
        out_channel=64,
        state_force_mlp_size=(64, 64), state_force_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type='pointnet',
        use_contact_forces = False, 
        # contact_mlp_size=(64, 64), contact_mlp_activation_fn=nn.ReLU,
        ):
        super().__init__()
        # self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        # self.rgb_image_key = 'image'

        self.contact_forces_key = 'contact_forces'

        
        # self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if use_contact_forces:
            self.contact_shape = observation_space[self.contact_forces_key]
            self.state_contact_shape = self.state_shape[0] + self.contact_shape[0]
        
        else: 
            self.state_contact_shape = self.state_shape[0]

        # if self.use_imagined_robot:
        #     self.imagination_shape = observation_space[self.imagination_key]
        # else:
        self.imagination_shape = None
            
    
        cprint(f"[CFMEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CFMEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[CFMEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        cprint(f"[PC ENCODER CFG]pointcloud_encoder_cfg: {pointcloud_encoder_cfg}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.use_contact_forces = use_contact_forces
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                nn_channels=[pointcloud_encoder_cfg.in_channels, 16, 32, 64]
                # self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                self.extractor = PointNetfeat(
                    nn_channels=nn_channels,
                )
            else:
                pointcloud_encoder_cfg.in_channels = 3
                nn_channels=[pointcloud_encoder_cfg.in_channels, 16, 32, 64]
                self.extractor = PointNetfeat(
                    nn_channels = nn_channels,
                )
                # self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        
        out_channel = nn_channels[-1]
        self.n_output_channels = out_channel

        assert len(state_force_mlp_size) >= 2
        # else:

        # concat joint_state and contact_mlp together
        net_arch = state_force_mlp_size[:-1]
        output_dim = state_force_mlp_size[-1]

        self.n_output_channels  += output_dim  # set to 64 in defalt params
        self.state_force_mlp = nn.Sequential(*create_mlp(self.state_contact_shape, output_dim, net_arch, state_force_mlp_activation_fn))

        cprint(f"[CFMEncoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[CFMEncoder] use_contact_forces: {use_contact_forces}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:  # false in isaac cube case
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))

        '''
        [CFMEncoder] points shape: torch.Size([2, 1024, 6])
        [CFMEncoder] pn_feat shape: torch.Size([2, 64])
        [CFMEncoder] state.shape: torch.Size([2, 24])
        [CFMEncoder] state_feat shape: torch.Size([2, 64])
        '''
        points = torch.transpose(points, 1, 2)   # B * 3 * N
        pointwise_feat, pn_feat, _, _ = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        contact = observations[self.contact_forces_key]

        state_force = torch.cat([state, contact], dim=-1)
        state_force_feat = self.state_force_mlp(state_force)  # B * 64
        final_feat = torch.cat([pn_feat, state_force_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels

class PointNetTactileLargeEncoder(nn.Module):
    def __init__(self, 
        observation_space: Dict, 
        # img_crop_shape=None,
        out_channel=64,
        state_force_mlp_size=(64, 64), state_force_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type='pointnet',
        use_contact_forces = False, 
        # contact_mlp_size=(64, 64), contact_mlp_activation_fn=nn.ReLU,
        ):
        super().__init__()
        # self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        # self.rgb_image_key = 'image'

        self.contact_forces_key = 'contact_forces'

        
        # self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if use_contact_forces:
            self.contact_shape = observation_space[self.contact_forces_key]
            self.state_contact_shape = self.state_shape[0] + self.contact_shape[0]
        
        else: 
            self.state_contact_shape = self.state_shape[0]

        # if self.use_imagined_robot:
        #     self.imagination_shape = observation_space[self.imagination_key]
        # else:
        self.imagination_shape = None
            
    
        cprint(f"[PointNetTactilelargeEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[PointNetTactilelargeEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[PointNetTactilelargeEncoder] imagination point shape: {self.imagination_shape}", "yellow")

        cprint(f"[PC ENCODER CFG]pointcloud_encoder_cfg: {pointcloud_encoder_cfg}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.use_contact_forces = use_contact_forces
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                nn_channels=[pointcloud_encoder_cfg.in_channels, 64, 128, 256]
                # self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                self.extractor = PointNetfeat(
                    nn_channels=nn_channels,
                )
            else:
                pointcloud_encoder_cfg.in_channels = 3
                nn_channels=[pointcloud_encoder_cfg.in_channels, 64, 128, 256]
                self.extractor = PointNetfeat(
                    nn_channels = nn_channels,
                )
                # self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        
        out_channel = nn_channels[-1]
        self.n_output_channels = out_channel

        assert len(state_force_mlp_size) >= 2
        # else:

        # concat joint_state and contact_mlp together
        net_arch = state_force_mlp_size[:-1]
        output_dim = state_force_mlp_size[-1]

        self.n_output_channels  += output_dim  # set to 64 in defalt params
        self.state_force_mlp = nn.Sequential(*create_mlp(self.state_contact_shape, output_dim, net_arch, state_force_mlp_activation_fn))

        cprint(f"[CFMEncoder] output dim: {self.n_output_channels}", "red")
        cprint(f"[CFMEncoder] use_contact_forces: {use_contact_forces}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:  # false in isaac cube case
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))

        '''
        [CFMEncoder] points shape: torch.Size([2, 1024, 6])
        [CFMEncoder] pn_feat shape: torch.Size([2, 64])
        [CFMEncoder] state.shape: torch.Size([2, 24])
        [CFMEncoder] state_feat shape: torch.Size([2, 64])
        '''
        points = torch.transpose(points, 1, 2)   # B * 3 * N
        pointwise_feat, pn_feat, _, _ = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        contact = observations[self.contact_forces_key]

        state_force = torch.cat([state, contact], dim=-1)
        state_force_feat = self.state_force_mlp(state_force)  # B * 64
        final_feat = torch.cat([pn_feat, state_force_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels