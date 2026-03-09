from typing import Dict
import torch
import torch.nn as nn
from conditional_flow_matching.model.common.module_attr_mixin import ModuleAttrMixin
from conditional_flow_matching.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()
    
    # a sub_function called by "predict_action", will be implemented in the derived class
    def conditional_flow_matching(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None, inference_step=5,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
