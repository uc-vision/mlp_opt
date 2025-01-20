def log_adam_behavior_to_wandb(image_id, gaussians, adam_optimizer, iter, rendered_image, psnr, loss):
    """
    Logs Adam optimizer behavior to wandb, associating logs with a specific dataset image.
    
    Args:
        image_id (str): A unique identifier for the dataset image being processed.
        gaussians: The model parameters being optimized.
        adam_optimizer: The optimizer used for training.
        iter (int): Current training iteration.
        rendered_image (torch.Tensor): The rendered image associated with this iteration.
        psnr (float): Peak Signal-to-Noise Ratio for the rendered image.
        loss (float): Loss value at the current iteration.
    """
    log_data = {
        "iteration": iter,
        "image_id": image_id,
        f"image_{image_id}/iter_{iter}/psnr": psnr,
        f"image_{image_id}/iter_{iter}/loss": loss,
    }

    # Iterate over parameter groups to log weights and biases
    for idx, param_group in enumerate(adam_optimizer.param_groups):
        for param in param_group['params']:
            if param.grad is not None:
                if param.ndimension() > 1:  # Likely weights
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/weights_mean"] = param.data.mean().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/weights_std"] = param.data.std().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/weights_grad_mean"] = param.grad.mean().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/weights_grad_std"] = param.grad.std().item()
                elif param.ndimension() == 1:  # Likely biases
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/biases_mean"] = param.data.mean().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/biases_std"] = param.data.std().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/biases_grad_mean"] = param.grad.mean().item()
                    log_data[f"image_{image_id}/iter_{iter}/param_group_{idx}/biases_grad_std"] = param.grad.std().item()
    
    # Log the rendered image associated with this dataset image
    log_data[f"image_{image_id}/iter_{iter}/rendered_image"] = wandb.Image(
        rendered_image.cpu().numpy(), caption=f"Rendered Image for Image {image_id} at Iteration {iter}"
    )

    # Log to wandb
    wandb.log(log_data)
