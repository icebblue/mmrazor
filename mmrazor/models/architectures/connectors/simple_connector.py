from typing import Tuple
import torch
import torch.nn as nn

from mmengine import MMLogger
from mmrazor.registry import MODELS
from .base_connector import BaseConnector

@MODELS.register_module()
class SimConnector(BaseConnector):
    def __init__(self,) -> None:
        super().__init__()
        
    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        logger = MMLogger.get_current_instance()
        logger.info(f'feature shape: {x.shape}.')
        return x