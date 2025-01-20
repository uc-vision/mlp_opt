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

from taichi_splatting.torch_lib.util import check_finite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--iters', type=int, default=20)

    parser.add_argument('--epoch',
                        type=int,
                        default=4,
                        help='base epoch size (increases with t)')
    # parser.add_argument('--wandb_run_id', type=str, required=True, help='wandb run ID to load parameters from')
    # parser.add_argument('--project_name', type=str, required=True, help='project name to load parameters from wandb')
    parser.add_argument('--max_epoch', type=int, default=16)
    parser.add_argument('--opacity_reg', type=float, default=0.0000)
    parser.add_argument('--scale_reg', type=float, default=10.0)
    parser.add_argument('--batch', type=int,default=1, help = "enable the batch size training default is 1")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--eval', action='store_false', help = "enable the evaluation phase during training default at every 5 images, can change the number with --test x")
    parser.add_argument('--test', type=int, default=5, help = "run the test phase at every x images default 5")
    parser.add_argument('--profile', action='store_true')
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log the gaussian and optimiser parameter to wandb')
    parser.add_argument('--method', type=str, default="mlp", help = "default mlp, other option mlp_unet"),
    args = parser.parse_args()
    return args


def log_lerp(t, a, b):
    return math.exp(math.log(b) * t + math.log(a) * (1 - t))


def lerp(t, a, b):
    return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()
    image = cv2.resize(image,(400*2,300*2))
    cv2.imshow(name, image)
    cv2.waitKey(1)


def psnr(a, b):
    return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))


def cat_values(dicts, dim=1):
    assert all(d.batch_size == dicts[0].batch_size for d in dicts)

    dicts = [d.to_tensordict() for d in dicts]
    assert all(d.keys() == dicts[0].keys() for d in dicts)

    keys, cls = dicts[0].keys(), type(dicts[0])
    concatenated = {k: torch.cat([d[k] for d in dicts], dim=dim) for k in keys}
    return cls.from_dict(concatenated, batch_size=dicts[0].batch_size)


def element_sizes(t):
    """ Get non batch sizes from a tensorclass"""
    return {k: v.shape[1:] for k, v in t.items()}


def split_tensorclass(t, flat_tensor: torch.Tensor):
    sizes = element_sizes(t)

    splits = [int(np.prod(s)) for s in sizes.values()]

    

    tensors = torch.split(flat_tensor, splits, dim=1)

    return t.__class__.from_dict(
        {
            k: v.view(t.batch_size + s)
            for k, v, s in zip(sizes.keys(), tensors, sizes.values())
        },
        batch_size=t.batch_size)


def flatten_tensorclass(t):
    flat_tensor = torch.cat([v.view(v.shape[0], -1) for v in t.values()],
                            dim=1)
    return flat_tensor


def mean_dicts(dicts: List[Dict[str, float]]):
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0].keys()}
