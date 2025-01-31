import os
import matplotlib.pyplot as plt
from functools import partial
import math
from typing import Dict, List
import cv2
import argparse
import numpy as np
import torch.nn as nn
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from mlp import mlp
from taichi_splatting.misc.renderer2d import project_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from utils import parse_args, log_lerp,lerp,mean_dicts,psnr,display_image
from taichi_splatting.torch_lib.util import check_finite
import os
import matplotlib.pyplot as plt
import torch.nn.utils as nn_utils



def element_sizes_batch(t):
    """ Get non batch sizes from a tensorclass"""

    return {k: v.shape[1:] for k, v in t.items()}

def split_tensorclass(t, flat_tensor: torch.Tensor):
    sizes = element_sizes_batch(t)
    splits = [int(np.prod(s)) for s in sizes.values()]
 
    tensors = torch.split(flat_tensor, splits, dim=2)
  
    k =  t.__class__.from_dict(
        {
            k: v.view(torch.Size([2, 5000])+s )  # Reshape tensor `v`
            for k, v, s in zip(sizes.keys(), tensors, sizes.values(
            ))  
        },
        batch_size=torch.Size([2, 5000]))# Set batch size for the new tensorclass)
    return k


def batch_images(image_files, batch_size):
    """Split image file paths into batches."""
    for i in range(0, len(image_files), batch_size):
        yield image_files[i:i + batch_size]
def load_batch_images(image_batch, device):
    """Load and preprocess a batch of images."""
    images = []
    for img_path in image_batch:
        img = cv2.imread(img_path)
        assert img is not None, f"Could not read {img_path}"
        img = torch.from_numpy(img).to(dtype=torch.float32, device=device) / 255
        images.append(img)
    

    return images

def psnr_batch_efficient(batch_a, batch_b):
    """
    Calculate PSNR for a batch of images more efficiently.

    Args:
    - batch_a: Tensor of shape [batch_size, channels, height, width] representing predicted images.
    - batch_b: Tensor of shape [batch_size, channels, height, width] representing reference images.

    Returns:
    - Average PSNR for the batch.
    """
    # Calculate MSE for each image in the batch
    mse = torch.nn.functional.mse_loss(batch_a, batch_b, reduction='none')
    mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)

    # Compute PSNR for each image in the batch
    psnr_per_image = 10 * torch.log10(1 / mse_per_image)

    # Return the average PSNR across the batch
    return psnr_per_image.mean()



def initialize_gaussians(batch_size, image_size, n_gaussians, device):
    """
    Initialize random Gaussians for a batch, ensuring each batch entry has unique parameters.
    """
    gaussians = []
    for _ in range(batch_size):
        # Randomly generate Gaussians for each batch entry
        single_gaussian = random_2d_gaussians(n_gaussians, image_size, alpha_range=(0.5, 1.0), scale_factor=1.0)
        gaussians.append(single_gaussian)

    # Stack into a single tensor for batched processing
    batched_gaussians = torch.stack(gaussians).to(device)  # Shape: [batch_size, n_gaussians, feature_size]
    return batched_gaussians



def flatten_tensorclass(t):
    flat_tensor = torch.cat(
        
        [v.view(v.shape[0], v.shape[1], -1) for v in t.values()], dim=2)
    
    return flat_tensor