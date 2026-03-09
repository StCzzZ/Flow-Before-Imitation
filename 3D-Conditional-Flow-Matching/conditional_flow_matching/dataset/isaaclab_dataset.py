from typing import Dict
import torch
import numpy as np
import copy
from conditional_flow_matching.common.pytorch_util import dict_apply
from conditional_flow_matching.common.replay_buffer import ReplayBuffer
from conditional_flow_matching.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from conditional_flow_matching.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from conditional_flow_matching.dataset.base_dataset import BaseDataset
from termcolor import cprint

class IsaaclabDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_contact_forces=False,
            ):
        super().__init__()
        self.task_name = task_name
        keys = ['agent_pos', 'action', 'point_cloud', 'state', 'obs']
        cprint(f"[IsaaclabDataset] use_contact_forces: {use_contact_forces}", 'green')
        if use_contact_forces:
            keys.append('contact_forces')
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_contact_forces = use_contact_forces

        # if self.use_contact_forces and self.replay_buffer['contact_forces'].shape[-1] == 3:
        #     # apply norm to the last dim
        #     self.replay_buffer['contact_forces'] = np.linalg.norm(self.replay_buffer['contact_forces'], axis=-1, keepdims=True)


        
        cprint(f"[IsaaclabDataset] episode_ends: {self.replay_buffer.root['meta']['episode_ends']}", 'green')
        cprint(f"[IsaaclabDataset] zarr_path: {zarr_path}", 'green')


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['agent_pos'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        if self.use_contact_forces:
            pass  #  we don't need to normalize the contact force for now, because we just need a binary contact
        #     data['contact_forces'] = self.replay_buffer['contact_forces']
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)
        if self.use_contact_forces:
            contact_forces = sample['contact_forces'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024 or 512, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        if self.use_contact_forces:
            data['obs']['contact_forces'] = contact_forces  # T, D_contact_forces
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

