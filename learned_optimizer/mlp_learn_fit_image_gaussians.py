from dataclasses import replace
from functools import partial
import math
import os
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import taichi as ti
import json
import torch
import numpy as np
from gaussian_mixer import GaussianMixer
from tqdm import tqdm
from utils import parse_args, partial, log_lerp, psnr, display_image, flatten_tensorclass, split_tensorclass, mean_dicts, lerp
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import point_basis, project_gaussians2d, uniform_split_gaussians2d
import wandb
from taichi_splatting.optim.fractional import FractionalAdam, SparseAdam, SparseLaProp
from taichi_splatting.optim.visibility_aware import VisibilityAwareAdam, VisibilityAwareLaProp, VisibilityOptimizer
from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile_size', type=int, default=16)
    parser.add_argument('--pixel_tile',
                        type=str,
                        help='Pixel tile for backward pass default "2,2"')

    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--target', type=int, default=None)
    parser.add_argument('--prune',
                        action='store_true',
                        help='enable pruning (equivalent to --target=n)')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--max_lr', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=0.1)

    parser.add_argument('--epoch',
                        type=int,
                        default=4,
                        help='base epoch size (increases with t)')
    parser.add_argument('--max_epoch', type=int, default=16)

    parser.add_argument(
        '--prune_rate',
        type=float,
        default=0.04,
        help='Rate of pruning proportional to number of points')
    parser.add_argument('--opacity_reg', type=float, default=0.0001)
    parser.add_argument('--scale_reg', type=float, default=10.0)

    parser.add_argument('--threaded',
                        action='store_true',
                        help='Use taichi dedicated thread')
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log the gaussian and optimiser parameter to wandb')

    parser.add_argument('--antialias', action='store_true')

    parser.add_argument('--write_frames', type=Path, default=None)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--profile', action='store_true')

    args = parser.parse_args()

    if args.pixel_tile:
        args.pixel_tile = tuple(map(int, args.pixel_tile.split(',')))

    return args


def log_lerp(t, a, b):
    return math.exp(math.log(b) * t + math.log(a) * (1 - t))


def lerp(t, a, b):
    return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)


def psnr(a, b):
    return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))


def train_epoch(opt: FractionalAdam,
                mlp_opt: torch.optim.Optimizer,
               mlp : torch.nn.Module,
                params:ParameterClass,
                ref_image,
                config: RasterConfig,
                epoch_size=100,
                grad_alpha=0.9,
                opacity_reg=0.0,
                scale_reg=0.0,
                loggable=False,
                iteration = 0):

    h, w = ref_image.shape[:2]

    point_heuristics = torch.zeros((params.batch_size[0], 2),
                                   device=params.position.device)
    visibility = torch.zeros((params.batch_size[0]),
                             device=params.position.device)

    for i in range(epoch_size):
        opt.zero_grad()

        with torch.enable_grad():
            
            gaussians = Gaussians2D.from_tensordict(params.tensors)

            
            gaussians.z_depth.requires_grad_(True) 
            gaussians_clone = gaussians.clone().detach()
            gaussians2d = project_gaussians2d(gaussians)

            raster = rasterize(gaussians2d=gaussians2d,
                               depth=gaussians.z_depth.clamp(0, 1),
                               features=gaussians.feature,
                               image_size=(w, h),
                               config=config)

            depth_reg = 0.0 * gaussians.z_depth.sum()
            scale = torch.exp(gaussians.log_scaling) / min(w, h)
            loss = (torch.nn.functional.l1_loss(raster.image, ref_image) +
                    opacity_reg * gaussians.opacity.mean() +
                    scale_reg * scale.pow(2).mean() + depth_reg)

            loss.backward()

        check_finite(gaussians, 'gaussians')
        visible = (raster.visibility > 1e-8).nonzero().squeeze(1)


    
        if isinstance(opt, VisibilityOptimizer):
            opt.step(indexes=visible,
                     visibility=raster.visibility[visible],
                     basis=point_basis(gaussians[visible]))
        else:
            opt.step(indexes=visible, basis=point_basis(gaussians[visible]))

        params.replace(rotation=torch.nn.functional.normalize(
            params.rotation.detach()),
                       log_scaling=torch.clamp(params.log_scaling.detach(),
                                               min=-5,
                                               max=5))

        # point_heuristics *= raster.visibility.clamp(1e-8).unsqueeze(1).sqrt()
        visibility += raster.visibility
        point_heuristics += raster.point_heuristics


            
        mlp_opt.zero_grad()
        model_step = gaussians_clone - gaussians
        gaussians[:] = gaussians_clone
        # Flatten gradients for MLP input
        grad = flatten_tensorclass(gaussians.grad)
        

        with torch.enable_grad():
            # Predict step using MLP
            predicted_step = mlp(grad, gaussians,
                                                ref_image.shape[:2],
                                                config,
                                                ref_image)
            predicted_step = split_tensorclass(gaussians, predicted_step)
            
            

            # Compute supervised loss for MLP
            mlp_loss = torch.nn.functional.l1_loss(flatten_tensorclass(model_step), flatten_tensorclass(predicted_step))
            mlp_loss.backward()

        mlp_opt.step()
        gaussians -= predicted_step



    if loggable is True:
        log_adam_behavior_to_wandb(gaussians=gaussians,
                                   adam_optimizer=opt,
                                   iter=iteration+epoch_size,
                                   rendered_image=raster.image)
    return raster.image, point_heuristics


def log_adam_behavior_to_wandb(gaussians, adam_optimizer, iter,
                               rendered_image):
    """
    Logs Adam optimizer behavior to wandb.
    """
    log_data = {"iter": iter}

    # Iterate over parameter groups to log weights and biases
    for idx, param_group in enumerate(adam_optimizer.param_groups):
        for param in param_group['params']:
            if param.grad is not None:
                if param.ndimension() > 1:  # Likely weights
                    log_data[f"param_group_{idx}/weights_mean"] = param.data.mean().item()
                    log_data[f"param_group_{idx}/weights_std"] = param.data.std().item()
                    log_data[f"param_group_{idx}/weights_grad_mean"] = param.grad.mean().item()
                    log_data[f"param_group_{idx}/weights_grad_std"] = param.grad.std().item()
                elif param.ndimension() == 1:  # Likely biases
                    log_data[f"param_group_{idx}/biases_mean"] = param.data.mean().item()
                    log_data[f"param_group_{idx}/biases_std"] = param.data.std().item()
                    log_data[f"param_group_{idx}/biases_grad_mean"] = param.grad.mean().item()
                    log_data[f"param_group_{idx}/biases_grad_std"] = param.grad.std().item()
    log_data[f"iter_{iter}/rendered_image"] = wandb.Image(
        rendered_image.cpu().numpy(), caption=f"Rendered Image at Iteration {iter}"
    )
    # Log to wandb
    wandb.log(log_data)


def make_epochs(total_iters, first_epoch, max_epoch):
    iteration = 0
    epochs = []
    while iteration < total_iters:

        t = iteration / total_iters
        epoch_size = math.ceil(log_lerp(t, first_epoch, max_epoch))

        if iteration + epoch_size * 2 > total_iters:
            # last epoch can just use the extra iterations
            epoch_size = total_iters - iteration

        iteration += epoch_size
        epochs.append(epoch_size)

    return epochs


@beartype
def take_n(t: torch.Tensor, n: int, descending=False):
    """ Return mask of n largest or smallest values in a tensor."""
    idx = torch.argsort(t, descending=descending)[:n]

    # convert to mask
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask[idx] = True

    return mask


def randomize_n(t: torch.Tensor, n: int):
    """ Randomly select n of the largest values in a tensor using torch.multinomial"""
    probs = F.normalize(t, dim=0)
    mask = torch.zeros_like(t, dtype=torch.bool)

    if n > 0:
        selected_indices = torch.multinomial(probs, n, replacement=False)
        mask[selected_indices] = True

    return mask


def find_split_prune(n, target, n_prune, point_heuristics):
    prune_cost, densify_score = point_heuristics.unbind(dim=1)

    prune_mask = take_n(prune_cost, n_prune, descending=False)

    target_split = ((target - n) + n_prune)
    # split_mask = randomize_n(densify_score, min(target_split, n))
    split_mask = take_n(densify_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both


def split_prune(params: ParameterClass, t, target, prune_rate,
                point_heuristics):
    n = params.batch_size[0]

    split_mask, prune_mask = find_split_prune(
        n=n,
        target=target,
        n_prune=int(prune_rate * n * (1 - t)),
        # n_prune=int(prune_rate * n),
        point_heuristics=point_heuristics)

    to_split = params[split_mask]

    splits = uniform_split_gaussians2d(Gaussians2D.from_tensordict(
        to_split.tensors),
                                       random_axis=True)
    optim_state = to_split.tensor_state.new_zeros(to_split.batch_size[0], 2)

    # optim_state['position']['running_vis'][:] = to_split.tensor_state['position']['running_vis'].unsqueeze(1) * 0.5

    params = params[~(split_mask | prune_mask)]
    params = params.append_tensors(splits.to_tensordict(),
                                   optim_state.reshape(splits.batch_size))
    # params.replace(rotation = torch.nn.functional.normalize(params.rotation.detach()))

    return params, dict(split=split_mask.sum().item(),
                        prune=prune_mask.sum().item())


def main():
    torch.set_printoptions(precision=4, sci_mode=False)

    cmd_args = parse_args()
    device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    ref_image = cv2.imread(cmd_args.image_file)
    assert ref_image is not None, f'Could not read {cmd_args.image_file}'

    h, w = ref_image.shape[:2]

    TaichiQueue.init(arch=ti.cuda,
                     log_level=ti.INFO,
                     debug=cmd_args.debug,
                     device_memory_GB=0.1,
                     threaded=cmd_args.threaded)

    

    # print(f'Image size: {w}x{h}')

    if cmd_args.show:
        cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rendered', w, h)

    torch.manual_seed(cmd_args.seed)
    lr_range = (cmd_args.max_lr, cmd_args.min_lr)

    torch.manual_seed(cmd_args.seed)
    torch.cuda.random.manual_seed(cmd_args.seed)
    gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                    alpha_range=(0.5, 1.0),
                                    scale_factor=0.5).to(
                                        torch.device('cuda:0'))

    parameter_groups = dict(position=dict(lr=lr_range[0], type='local_vector'),
                            log_scaling=dict(lr=0.05),
                            rotation=dict(lr=1.0),
                            alpha_logit=dict(lr=0.1),
                            feature=dict(lr=0.025, type='vector'))

    # params = ParameterClass(gaussians.to_tensordict(),
    #       parameter_groups, optimizer=SparseAdam, betas=(0.9, 0.95), eps=1e-16, bias_correction=True)
    n_inputs = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])
    
    mlp = GaussianMixer(inputs=n_inputs,
                              outputs=n_inputs,
                              n_render=16,
                              n_base=128,
                              method = "mlp").to(device)
    mlp.to(device=device)


    mlp = torch.compile(mlp)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

    params = ParameterClass(gaussians.to_tensordict(),
                            parameter_groups,
                            optimizer=VisibilityAwareLaProp,
                            vis_beta=0.9,
                            betas=(0.9, 0.9),
                            eps=1e-16,
                            bias_correction=False)

    keys = set(params.keys())
    trainable = set(params.optimized_keys())

    print(f'attributes - trainable: {trainable} other: {keys - trainable}')

    ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32,
                                               device=device) / 255

    config = RasterConfig(
        compute_point_heuristics=True,
        compute_visibility=True,
        tile_size=cmd_args.tile_size,
        blur_cov=0.3 if not cmd_args.antialias else 0.0,
        antialias=cmd_args.antialias,
        # alpha_threshold=1/8192,
        pixel_stride=cmd_args.pixel_tile or (2, 2))

    def timed_epoch(*args, **kwargs):
        start = time.time()
        image, point_heuristics = train_epoch(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()

        return image, point_heuristics, end - start

    if cmd_args.wandb is True:
        wandb.init(project="gaussian2d-adam-behavior",
                   config={
                       "learning_rate": 0.001,
                       "architecture": "Gaussian2D",
                       "optimizer": "Adam",
                       "ref_image": cmd_args.image_file
                   })
    train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch
    epochs = make_epochs(cmd_args.iters, cmd_args.epoch, cmd_args.max_epoch)

    pbar = tqdm(total=cmd_args.iters)
    iteration = 0
    for epoch_size in epochs:

        t = (iteration + epoch_size * 0.5) / cmd_args.iters
        params.set_learning_rate(position=log_lerp(t, *lr_range))
        metrics = {}

        image, point_heuristics, epoch_time = train(
            params.optimizer,
            mlp_opt,
            mlp,
            params,
            ref_image,
            epoch_size=epoch_size,
            config=config,
            opacity_reg=cmd_args.opacity_reg,
            scale_reg=cmd_args.scale_reg,
            loggable=cmd_args.wandb,
            iteration = iteration
            )

        if cmd_args.show:
            display_image('rendered', image)

        if cmd_args.write_frames:
            filename = cmd_args.write_frames / f'{iteration:04d}.png'
            filename.parent.mkdir(exist_ok=True, parents=True)
            print(f'Writing {filename}')
            cv2.imwrite(str(filename),
                        (image.detach().clamp(0, 1) * 255).cpu().numpy())

        metrics['CPSNR'] = psnr(ref_image, image).item()
        metrics['n'] = params.batch_size[0]

        if cmd_args.prune and cmd_args.target is None:
            cmd_args.target = cmd_args.n

        if cmd_args.target and iteration + epoch_size < cmd_args.iters:
            t_points = min(math.pow(t * 2, 0.5), 1.0)
            target = math.ceil(params.batch_size[0] * (1 - t_points) +
                               t_points * cmd_args.target)
            params, prune_metrics = split_prune(params, t, target,
                                                cmd_args.prune_rate,
                                                point_heuristics)
            metrics.update(prune_metrics)

        for k, v in metrics.items():
            if isinstance(v, float):
                metrics[k] = f'{v:.2f}'
            if isinstance(v, int):
                metrics[k] = f'{v:4d}'

        pbar.set_postfix(**metrics)

        iteration += epoch_size
        pbar.update(epoch_size)


def with_benchmark(f):

    def g(*args, **kwargs):
        with profile(activities=[ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            with record_function("model_inference"):
                result = f(*args, **kwargs)
                torch.cuda.synchronize()

            prof_table = prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=25,
                max_name_column_width=100)
            print(prof_table)
            return result

    return g


main()
