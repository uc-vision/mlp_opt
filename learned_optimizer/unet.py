
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
from mlp import mlp_body, mlp
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
from utils import parse_args, log_lerp, psnr, display_image, flatten_tensorclass, split_tensorclass, mean_dicts, lerp
from fused_ssim import fused_ssim
from taichi_splatting.torch_lib.util import check_finite




def normalize_raster(raster_image: torch.Tensor,
                     raster_alpha: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """
    Normalizes the raster image using raster.image_weight
     with an epsilon for numerical stability.
    
    Args:
        raster_image (torch.Tensor): The raster image tensor (e.g., shape [B, C, H, W]).
        raster_alpha (torch.Tensor): The image_weight tensor for normalization (e.g., shape [B, 1, H, W]).
        eps (float): A small epsilon value to prevent division by zero.
        
    Returns:
        torch.Tensor: The normalized raster image.
    """
    m = nn.Sigmoid()
    normalized_image = raster_image / (m(raster_alpha).unsqueeze(-1) + eps)
    return normalized_image


def group_norm(num_channels: int):
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=False)


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):

    def __init__(self, in_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.theta_x = nn.Conv2d(in_channels,
                                 inter_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.phi_g = nn.Conv2d(in_channels,
                               inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.psi = nn.Conv2d(inter_channels,
                             1,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = self.relu(theta_x + phi_g)

        psi = self.sigmoid(self.psi(f))
        return x * psi


class UNet3(nn.Module):
    "This is the 3 layer Unet with pooling and attention gate"

    def __init__(self, dropout_rate=0.5, l2_lambda=0.01):
        super(UNet4, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Dropout(dropout_rate))

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2))

        self.layer1 = conv_block(19, 64, dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = conv_block(64, 128, dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = conv_block(128, 256, dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.layer4 = conv_block(256, 512, dropout_rate)

        # Expanding Path
        self.upconv3 = upconv_block(512, 256)
        self.attention3 = AttentionGate(256, 128)
        self.layer5 = conv_block(512, 256, dropout_rate)

        self.upconv2 = upconv_block(256, 128)
        self.attention2 = AttentionGate(128, 64)
        self.layer6 = conv_block(256, 128, dropout_rate)

        self.upconv1 = upconv_block(128, 64)
        self.attention1 = AttentionGate(64, 32)
        self.layer7 = conv_block(128, 19, dropout_rate)

        # Output Layer
        self.output_layer = nn.Conv2d(19, 19, kernel_size=1)

    def forward(self, x):

        x1 = self.layer1(x)
        p1 = self.pool1(x1)

        x2 = self.layer2(p1)
        p2 = self.pool2(x2)

        x3 = self.layer3(p2)
        p3 = self.pool3(x3)

        x4 = self.layer4(p3)

        up3 = self.upconv3(x4)

        att3 = self.attention3(x3, up3)

        concat3 = torch.cat([up3, att3], dim=1)
        x5 = self.layer5(concat3)

        up2 = self.upconv2(x5)
        att2 = self.attention2(x2, up2)
        concat2 = torch.cat([up2, att2], dim=1)
        x6 = self.layer6(concat2)

        up1 = self.upconv1(x6)

        att1 = self.attention1(x1, up1)
        concat1 = torch.cat([up1, att1], dim=1)
        x7 = self.layer7(concat1)

        output = self.output_layer(x7)

        return output


class UNet2D(nn.Module):
    "Unet with the down and up layer"

    def __init__(self,
                 f: int,
                 activation: Callable = nn.ReLU,
                 norm: Callable = group_norm) -> None:
        super().__init__()

        # Downsampling: f -> 2f -> 4f -> 8f -> 16f
        def make_down_layer(i: int) -> nn.Sequential:
            in_channels = f * (2**i)
            out_channels = 2 * in_channels
            return nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                          stride=2),
                norm(out_channels),
                activation(),
            )

        self.down_layers = nn.ModuleList(
            [make_down_layer(i) for i in range(2)])

        # Up path: 16f -> 8f -> 4f -> 2f -> f
        def make_up_layer(i: int) -> nn.Sequential:
            out_channels = f * (2**i)
            in_channels = 2 * out_channels
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2),
                norm(out_channels),
                activation(),
            )

        self.up_layers = nn.ModuleList(
            reversed([make_up_layer(i) for i in range(2)]))

        self.final_layer = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1),
            norm(f),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediates: List[torch.Tensor] = []
        # Down path: f -> 2f -> 4f -> 8f -> 16f
        for layer in self.down_layers:
            intermediates.append(x)
            x = layer(x)

        # Up path with skip connections: 16f -> 8f -> 4f -> 2f -> f
        intermediates = list(reversed(intermediates))

        for i, layer in enumerate(self.up_layers):
            x = layer(x) + intermediates[i]

        return self.final_layer(x)


class UNet2D_Model(nn.Module):

    def __init__(self,
                 inputs: int,
                 outputs: int,
                 n_render: int = 16,
                 n_base: int = 64,
                 ):
        super().__init__()
        self.down_project = nn.Linear(n_base, n_render)
        # Increase input size to include rendering features
        self.init_mlp = mlp_body(inputs,
                                 hidden_channels=[n_base] * 2,
                                 activation=nn.ReLU,
                                 norm=nn.LayerNorm)
        self.up_project = nn.Linear(n_render + 3 , n_base)
        self.unet = UNet2D(f=n_render + 3 , activation=nn.ReLU)

        self.final_mlp = mlp(n_base,
                            outputs=outputs,
                            hidden_channels=[n_base] * 2,
                            activation=nn.ReLU,
                            norm=nn.LayerNorm,
                            output_scale=1e-12)


    @torch.compiler.disable
    def render(self,
               features: torch.Tensor,
               gaussians: Gaussians2D,
               image_size: Tuple[int, int],
               raster_config: RasterConfig = RasterConfig()):
        h, w = image_size

        gaussians2d = project_gaussians2d(gaussians)
        raster = rasterize(gaussians2d=gaussians2d,
                           depth=gaussians.z_depth.clamp(0, 1),
                           features=features,
                           image_size=(w, h),
                           config=raster_config)
        raster_image = normalize_raster(raster_image=raster.image,
                                        raster_alpha=raster.image_weight,
                                        eps=1e-12)
        return raster_image.unsqueeze(0).permute(0, 3, 1, 2).to(
            memory_format=torch.channels_last
        )  # B, H, W, n_render -> 1, n_render, H, W

    def sample_positions(
            self,
            image: torch.Tensor,  # 1, n_render, H, W
            positions: torch.Tensor,  # B, 2
    ) -> torch.Tensor:  # B, n_render
        h, w = image.shape[-2:]
        B = positions.shape[0]
        # normalize positions to be in the range [w, h] -> [-1, 1] for F.grid_sample
        positions = ((positions / positions.new_tensor([w, h])) * 2.0 -
                     1.0).view(1, 1, B, 2)
        samples = F.grid_sample(image, positions,
                                align_corners=False)  # B, n_render, 1, 1
        return samples.view(B, -1)  # B, n_render, 1, 1 -> B, n_render

    def forward(self, x: torch.Tensor, gaussians: Gaussians2D,
                image_size: Tuple[int, int], raster_config: RasterConfig,
                ref_image: torch.Tensor) -> torch.Tensor:
    
        feature = self.init_mlp(x)      # B,inputs -> B, n_base
        x = self.down_project(feature)  # B, n_base -> B, n_render

        image = self.render(x.to(torch.float32), gaussians, image_size, raster_config) # B, n_render -> 1, n_render, H, W

        precon_image = ref_image.unsqueeze(0).permute(0,3,1,2)
        con_image = torch.cat((precon_image,image),dim=1)

        image = self.unet(con_image)   # B, n_render, H, W -> B, n_render, H, W

        # sample at gaussian centres from the unet output
        x = self.sample_positions(image, gaussians.position)
        x = self.up_project(x)              # B, n_render -> B, n_base
        # shortcut from output of init_mlp
        x = self.final_mlp(x+ feature)  # B, n_base -> B, outputs

        return x
