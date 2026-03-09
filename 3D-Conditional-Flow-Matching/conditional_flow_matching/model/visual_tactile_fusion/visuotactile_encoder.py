from conditional_flow_matching.model.visual_tactile_fusion.point_feature_extractor import PointFeatureMLPXYZRGB, PointFeatureMLPXYZTactile, PointNetEncoderXYZ, PointNetEncoderXYZRGB, create_mlp
from conditional_flow_matching.model.visual_tactile_fusion.transformer_fusion import TransformerFusion
from typing import Optional, Dict, Tuple, Union, List, Type

import torch
import torch.nn as nn
from termcolor import cprint



class VisualTactileEncoder(nn.Module):
    def __init__(self,
                 point_feature_extractor_cfg=None,
                 tactile_feature_extractor_cfg=None,
                 transformer_fusion_cfg=None,
                 out_channels: int=64,
                 final_norm: str='none',
                 ):
        
        super().__init__()
        
        if point_feature_extractor_cfg:
            self.point_feature_extractor = PointFeatureMLPXYZRGB(
                **point_feature_extractor_cfg
            )

        if tactile_feature_extractor_cfg:
            self.tactile_feature_extractor =PointFeatureMLPXYZTactile(
                **tactile_feature_extractor_cfg
            )
        
        if tactile_feature_extractor_cfg:
            self.transformer_fusion = TransformerFusion(
                **transformer_fusion_cfg
            )

        cprint(f"[VisualTactileEncoder] transformer_fusion_cfg.fea_channels: {transformer_fusion_cfg.fea_channels}", "yellow")
        cprint(f"[VisualTactileEncoder] transformer_fusion_cfg.fea_channels[-1]: {transformer_fusion_cfg.fea_channels[-1]}", "yellow")
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(transformer_fusion_cfg.fea_channels[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(transformer_fusion_cfg.fea_channels[-1], out_channels)

    def forward(self, point_cloud_xyz, point_cloud_all, tactile_data_xyz, tactile_data_all):
        point_features = self.point_feature_extractor(point_cloud_all)
        tactile_features = self.tactile_feature_extractor(tactile_data_all)

        fused_features = self.transformer_fusion(
            search_feature = tactile_features,
            search_coord = tactile_data_xyz,        
            template_feature=point_features,
            template_coord=point_cloud_xyz
        )
        # max pooling
        fused_features = torch.max(fused_features, 1)[0]
        # final projection
        fused_features = self.final_projection(fused_features)      
        return fused_features


class CFMRichContactEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 point_cloud_out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 contact_encoder_outdim=64,
                 contact_encoder_final_norm='none',
                 point_feature_extractor_cfg=None,
                 tactile_feature_extractor_cfg=None,
                 transformer_fusion_cfg=None
                 ):
        super().__init__()
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.tactile_point_all_keys = 'hand_contact_point_all'
        self.n_output_channels = point_cloud_out_channel
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        cprint(f"[CFMEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CFMEncoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[PC ENCODER CFG]pointcloud_encoder_cfg: {pointcloud_encoder_cfg}", "yellow")
        
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
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

        self.contact_encoder = VisualTactileEncoder(
            point_feature_extractor_cfg=point_feature_extractor_cfg,
            tactile_feature_extractor_cfg=tactile_feature_extractor_cfg,
            transformer_fusion_cfg=transformer_fusion_cfg,
            out_channels=contact_encoder_outdim,
            final_norm=contact_encoder_final_norm,
        )
        self.n_output_channels += contact_encoder_outdim

        cprint(f"[CFMEncoder] output dim: {self.n_output_channels}", "red")
    

    def forward(self, observations:Dict)->torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"points shape must be 3 {points.shape}", "red")
        pc_feat = self.extractor(points) # B, out_channel
        

        state = observations[self.state_key]
        state_feat = self.state_mlp(state) # B, 64


        tactile_data = observations[self.tactile_point_all_keys]  # B*T, 456, 4
        assert(len(tactile_data.shape) == 3), cprint(f"tactile_data shape must be 3 {tactile_data.shape}", "red")
        tactile_data_xyz = tactile_data[..., :3]
        tactile_data_all = tactile_data
        points_xyz = points[..., :3]
        points_all = points

        visuotactile_feat = self.contact_encoder(points_xyz, points_all, tactile_data_xyz, tactile_data_all)

        final_feat = torch.cat([pc_feat, state_feat, visuotactile_feat], dim=1)

        return final_feat

    def output_shape(self):
        return self.n_output_channels

        
