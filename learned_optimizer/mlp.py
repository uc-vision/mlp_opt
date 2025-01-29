from typing import List, Tuple
import torch
import torch.nn as nn

from taichi_splatting.misc.renderer2d import point_covariance
from renderer2d import Gaussians2D
import taichi as ti

from typing import Callable, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from renderer2d import Gaussians2D
import wandb
from taichi_splatting.rasterizer.function import rasterize

import os
import matplotlib.pyplot as plt
from functools import partial
import math
import cv2
import argparse
import numpy as np

import torch
from tqdm import tqdm
from fused_ssim import fused_ssim
from taichi_splatting.torch_lib.util import check_finite


def linear(in_features, out_features,  init_std=None):
  m = nn.Linear(in_features, out_features, bias=True)

  if init_std is not None:
    m.weight.data.normal_(0, init_std)
    
  m.bias.data.zero_()
  return m


def layer(in_features, out_features, activation=nn.Identity, norm=nn.Identity, **kwargs):
  return nn.Sequential(linear(in_features, out_features, **kwargs), 
                       norm(out_features),
                       activation(),
                       )


def mlp_body(inputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity):
  return nn.Sequential(
    layer(inputs, hidden_channels[0], activation),
    *[layer(hidden_channels[i], hidden_channels[i+1], activation, norm)  
      for i in range(len(hidden_channels) - 1)]
  )   


def mlp(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity, 
        output_activation=nn.Identity, output_scale =None):

  output_layer = layer(hidden_channels[-1], outputs, 
                       output_activation,
                       init_std=output_scale)
  
  return nn.Sequential(
    mlp_body(inputs, hidden_channels, activation, norm),
    output_layer
  )   


class MLP_Model(nn.Module):

    def __init__(self,
                 inputs: int,
                 outputs: int,
                 n_render: int = 16,
                 n_base: int = 64,
                 ):
        super().__init__() 
        
        self.final_mlp = mlp(inputs,
                            outputs=outputs,
                            hidden_channels=[n_base] * 2,
                            activation=nn.ReLU,
                            norm=nn.LayerNorm,
                            output_scale=1e-12)



    def forward(self, x: torch.Tensor,
                image_size: Tuple[int, int], raster_config: RasterConfig,
                ref_image: torch.Tensor) -> torch.Tensor:
        
        x = self.final_mlp(x)
        return x