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
from Batch_training import load_batch_images
from tqdm import tqdm
from utils import parse_args, partial, log_lerp, psnr, display_image, mean_dicts, lerp
from Batch_training import flatten_tensorclass, split_tensorclass
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
from taichi_splatting.examples.fit_image_gaussians import parse_args, make_epochs , train_epoch
import time
import torch.nn.functional as F
from mlp import MLP_Model

class ParamPair:
    def __init__(self, params,ref_image):
        self.params = params
        self.ref_image = ref_image
        self.param_before = None
        self.gaussian_image= None
        self.metrics = None
        self.gaussian = None
        self.raster = None
        self.mlp_image = None

class Trainer:
    def __init__(self, mlp,mlp_opt,config, ref_image, params_list):
        self.config = config
        self.ref_image = ref_image
        self.params_list = params_list
        self.mlp = mlp
        self.mlp_opt = mlp_opt

    def train_epoch(self,  
        params,
        ref_image,
        epoch_size=100, 
        grad_alpha=0.9, 
        opacity_reg=0.0,
        scale_reg=0.0):
    
        h, w = ref_image.shape[:2]

        point_heuristic = torch.zeros((params.batch_size[0], 2), device=params.position.device)
        visibility = torch.zeros((params.batch_size[0]), device=params.position.device)
   
        for i in range(epoch_size):
            params.optimizer.zero_grad()

            with torch.enable_grad():
                gaussians = Gaussians2D.from_tensordict(params.tensors)
                gaussians2d = project_gaussians2d(gaussians)  

                raster = rasterize(gaussians2d=gaussians2d, 
                    depth=gaussians.z_depth.clamp(0, 1),
                    features=gaussians.feature, 
                    image_size=(w, h), 
                    config=self.config)
                
                image = raster.image.sigmoid()

                
                scale = torch.exp(gaussians.log_scaling) / min(w, h)
                loss = (torch.nn.functional.mse_loss(image, ref_image) 
                        + opacity_reg * gaussians.opacity.mean()
                        + scale_reg * scale.pow(2).mean())

                loss.backward()


            check_finite(gaussians, 'gaussians')
            visibility = raster.visibility
            visible = (visibility > 1e-8).nonzero().squeeze(1)



            if isinstance(params.optimizer, VisibilityOptimizer):
                params.optimizer.step(indexes = visible, 
                        visibility=visibility[visible], 
                        basis=point_basis(gaussians[visible]))
            else:
                params.optimizer.step(indexes = visible, 
                        basis=point_basis(gaussians[visible]))

            params.replace(
            rotation = torch.nn.functional.normalize(params.rotation.detach()),
            log_scaling = torch.clamp(params.log_scaling.detach(), min=-5, max=5)
            )

            point_heuristic +=  raster.point_heuristic
            visibility += raster.visibility


        return image, (point_heuristic[:, 0], point_heuristic[:, 1]),params

    def train_orignal_model(self,epoch_size,grad_alpha,opacity_reg,scale_reg):
        for paramPair in self.params_list:

            paramPair.param_before = paramPair.params.tensors.clone().detach()

            paramPair.gaussian_image, metrics, paramPair.params = self.train_epoch(
                paramPair.params,
                paramPair.ref_image,
                epoch_size=epoch_size,
                grad_alpha=grad_alpha,
                opacity_reg=opacity_reg,
                scale_reg=scale_reg
            )


    def train(self, epoch_size=100, grad_alpha=0.9, opacity_reg=0.0, scale_reg=0.0):
    
        self.train_orignal_model(epoch_size,grad_alpha, opacity_reg,scale_reg)
        self.train_mlp(epoch_size,opacity_reg,scale_reg)
    
        return 0


    def render_step(self,epoch_size,
                opacity_reg,
                scale_reg):
        gradients = []
        batch_loss = 0
        for paramPair in self.params_list:
            
            h, w = paramPair.ref_image.shape[:2]
            gaussians = Gaussians2D.from_tensordict(paramPair.param_before)
            gaussians.requires_grad_(True)
            gaussians.z_depth.requires_grad_(True) 
            gaussians2d = project_gaussians2d(gaussians)  
            raster = rasterize(gaussians2d=gaussians2d, 
                depth=gaussians.z_depth.clamp(0, 1),
                features=gaussians.feature, 
                image_size=(w, h), 
                config=self.config)
            paramPair.raster = raster
            with torch.enable_grad():
                image = raster.image

                depth_reg = 0.0 * gaussians.z_depth.sum()
                scale = torch.exp(gaussians.log_scaling) / min(w, h)
                loss = (torch.nn.functional.mse_loss(image, paramPair.ref_image) 
                        + opacity_reg * gaussians.opacity.mean()
                        + scale_reg * scale.pow(2).mean() + depth_reg)
                batch_loss += loss
            paramPair.gaussian = gaussians


        with torch.enable_grad():
            batch_loss.backward()

        for paramPair in self.params_list:
            gradients.append(paramPair.gaussian.grad.clone().detach())
        return torch.stack(gradients)
       

    def render_gaussians(self, mlp_params):
        """
        Render the image based on updated Gaussian parameters.

        Args:
            mlp_params: Updated Gaussian parameters from MLP.

        Returns:
            Rendered image.
        """
        mlp_gaussians = Gaussians2D.from_tensordict(mlp_params)
        mlp_raster = rasterize(
            gaussians2d=project_gaussians2d(mlp_gaussians),
            depth=mlp_gaussians.z_depth.clamp(0, 1),
            features=mlp_gaussians.feature,
            image_size=self.params_list[0].ref_image.shape[:2],
            config=self.config
        )
        return mlp_raster

    def get_model_step(self):
        model_step =[]
        for ParamPair in self.params_list:
            model_step.append(ParamPair.param_before - ParamPair.params.tensors)
        return torch.stack(model_step)
    def train_mlp(self,epoch_size,opacity_reg,scale_reg):
        """
        Train the MLP to learn the parameter updates.

        Args:
            mlp: The MLP model.
            mlp_opt: Optimizer for the MLP.
            params: Gaussian parameters (before update).
            grad: Gradients of the parameters.

        Returns:
            The updated MLP loss.
        """
        
        self.mlp_opt.zero_grad()
        model_step = self.get_model_step()

        with torch.enable_grad():
            grad = self.render_step(epoch_size,opacity_reg,scale_reg)
            grad_flat = flatten_tensorclass(grad)
 
            predicted_step = self.mlp(grad_flat,
                                                self.params_list[0].ref_image.shape[:2],
                                                self.config,
                                                self.ref_image)
           
            predicted_step = split_tensorclass(self.params_list[0].gaussian, predicted_step)
           
            # Compute supervised loss for MLP
            mlp_loss = torch.nn.functional.l1_loss(flatten_tensorclass(model_step), flatten_tensorclass(predicted_step))
            mlp_loss.backward()

    
        self.mlp_opt.step()
        for i, paramPair in enumerate(self.params_list):
            param_step = predicted_step[i] 
            paramPair.params.tensors.replace(paramPair.param_before - param_step)
            paramPair.mlp_image = self.render_gaussians(paramPair.params.tensors).image.sigmoid()
        
        
def main_mlp():
    
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

    n_inputs = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()]) 
    
    mlp = MLP_Model(inputs=n_inputs,
                              outputs=n_inputs,
                              n_render=16,
                              n_base=128,
                            ).to(device)
    mlp.to(device=device)


    mlp = torch.compile(mlp)
    mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.001)

    batch_size = 2

    params = ParameterClass(gaussians.to_tensordict(),
                            parameter_groups,
                            optimizer=VisibilityAwareLaProp,
                            vis_beta=0.9,
                            betas=(0.9, 0.9),
                            eps=1e-16,
                            bias_correction=False)
    



    keys = set(params.keys())
    trainable = set(params.optimized_keys())

    ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32,
                                               device=device) / 255

    dataset_folder = "/csse/users/pwl25/pear/images"  # Using the argument as a folder path
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]


    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    config = RasterConfig(
        compute_point_heuristic=True,
        compute_visibility=True,
        tile_size=cmd_args.tile_size,
        blur_cov=0.3 if not cmd_args.antialias else 0.0,
        antialias=cmd_args.antialias,
        # alpha_threshold=1/8192,
        pixel_stride=cmd_args.pixel_tile or (2, 2))


    trainer = Trainer(mlp,mlp_opt, config, None, None)

    epochs = make_epochs(cmd_args.iters, cmd_args.epoch, cmd_args.max_epoch)

    pbar = tqdm(total=cmd_args.iters)
    
    iteration = 0
    for batch_i, batch_files in enumerate(batches, start=1):  
        params_list = [ParamPair(params, ref_image) for ref_image in load_batch_images(batch_files, device)]
        trainer.params_list = params_list

        for epoch_size in epochs:

            t = (iteration + epoch_size * 0.5) / cmd_args.iters
            l_r = log_lerp(t, *lr_range)
            for param_pair in trainer.params_list:
                param_pair.params.set_learning_rate(position=l_r)

            metrics = {}
            trainer.train(
                
                epoch_size=epoch_size,
                opacity_reg=cmd_args.opacity_reg,
                scale_reg=cmd_args.scale_reg,
                )

            if cmd_args.show:
                display_image('rendered', trainer.params_list[0].gaussian_image)
                display_image('original', trainer.params_list[0].ref_image)
                display_image('mlp', trainer.params_list[0].mlp_image)

            if cmd_args.write_frames:
                filename = cmd_args.write_frames / f'{iteration:04d}.png'
                filename.parent.mkdir(exist_ok=True, parents=True)
            
                cv2.imwrite(str(filename),
                            (image.detach().clamp(0, 1) * 255).cpu().numpy())

            metrics['CPSNR'] = psnr(trainer.params_list[0].ref_image, trainer.params_list[0].gaussian_image).item()
            metrics['MLP_CPSNR'] = psnr(trainer.params_list[0].ref_image, trainer.params_list[0].mlp_image).item()
            metrics['n'] = trainer.params_list[0].params.batch_size[0]

            if cmd_args.prune and cmd_args.target is None:
                cmd_args.target = cmd_args.n

            if cmd_args.target and iteration + epoch_size < cmd_args.iters:
                t_points = min(math.pow(t * 2, 0.5), 1.0)
                target = math.ceil(trainer.params.batch_size[0] * (1 - t_points) +
                                t_points * cmd_args.target)
                trainer.params, prune_metrics = split_prune(trainer.params, t, target,
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


if __name__ == "__main__":
    main_mlp()
