# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:45:59 2023

@author: pio-r
"""

import torch
from tqdm import tqdm
import torch.nn as nn
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion_cond:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, img_channel=1, device="cuda"):
        self.noise_steps = noise_steps # timestesps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alphas_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha[:-1]], dim=0)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_hat[:-1]], dim=0)
        self.sigma = torch.sqrt(1 - self.alpha_hat)  # Standard deviation of the noise at each timestep
        # self.alphas_cumprod_prev = torch.from_numpy(np.append(1, self.alpha_hat[:-1].cpu().numpy())).to(device)
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # linear variance schedule as proposed by Ho et al 2020

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ # equation in the paper from Ho et al that describes the noise processs

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sigma_to_timestep(self, sigma):
        # Find the index of the timestep where sigma is closest to the given value
        return torch.argmin(torch.abs(self.sigma - sigma)).item()
    
    def return_alphacumprof(self):
        return self.alpha_hat
    
    def return_sigma(self):
        return self.sigma

    def sample(self, model, n, y, labels, cfg_scale=3, eta=1, sampling_mode='ddpm'):
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, self.img_channel, self.img_size, self.img_size)).to(self.device)
            x = torch.cat([x, y], dim=1)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                
                predicted_noise = model(x, t)
                # if cfg_scale > 0:
                #     uncond_predicted_noise = model(x, y, None, t)
                #     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                 
                
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None] # this is noise, created in one
                alpha_prev = self.alphas_cumprod_prev[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                # SAMPLING adjusted from Stable diffusion
                sigma = (
                            eta
                            * torch.sqrt((1 - alpha_prev) / (1 - alpha_hat)
                            * (1 - alpha_hat / alpha_prev))
                        )
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train() # it goes back to training mode
        
        x = x + y
        x = (x.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        x = (x * 255).type(torch.uint8) # to bring in valid pixel range
        return x
    
    def sample_with_gradients(self, model, n, y, labels, cfg_scale=3, eta=1, sampling_mode='ddpm', steps=50):
        logging.info(f"Sampling {n} new images and storing gradients per timestep...")
        model.train()  # evaluation mode
    
        # Ensure y has requires_grad=True to compute gradients
        y.requires_grad_(True)
    
        # Initialize storage for gradients
        int_grad = []
    
        x = torch.randn((n, self.img_channel, self.img_size, self.img_size), requires_grad=True).to(self.device)
        
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):  # reverse loop from T to 1
            t = (torch.ones(n) * i).long().to(self.device)  # create timesteps tensor of length n
            
            baseline = -1*torch.ones_like(x)
            # Scale input and compute gradients
            scaled_inputs = [(baseline + (float(i) / steps) * (y - baseline)).requires_grad_(True) 
                             for i in range(0, steps + 1)]
            
            # Forward pass through the model
            x.requires_grad_(True)
            predicted_noise = model(x, y, labels, t)
            
            grads = []
            for value in tqdm(scaled_inputs):
                model.zero_grad()
                predicted_noise = model(x, value, labels, t)
                
                # grads.append(scaled_input.grad.detach().cpu().numpy())
            
                # Compute the gradient of `predicted_noise` with respect to `y`
                grad = torch.autograd.grad(
                    outputs=predicted_noise, 
                    inputs=value, 
                    grad_outputs=torch.ones_like(predicted_noise),
                    create_graph=False,  # Set to True if you need higher-order gradients
                    retain_graph=False,     # Retain graph for further gradient calculations
                )[0]
                
                grads.append(grad.detach().cpu().numpy())
    
            # Store the gradients for this timestep
            
            grads = np.array(grads)
            avg_grads = (grads[:-1] + grads[1:]) / 2.0 # average gradients at each step using the trapezoidal rule
            integrated_grads = (y.cpu().detach().numpy() - baseline.repeat(1, 5, 1, 1).cpu().detach().numpy()) * np.mean(avg_grads, axis=0)
            
            int_grad.append(integrated_grads)  # Detach and move to CPU to save memory
    
            # Sampling logic remains unchanged
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            alpha_prev = self.alphas_cumprod_prev[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
    
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_hat) * (1 - alpha_hat / alpha_prev))
            )
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
    
            pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    
        # Return the sampled images and the collected gradients
        return x, int_grad


mse = nn.MSELoss()

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
    r"""Create a function that calculates the PSNR between 2 images.

    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    Given an m x n image, the PSNR is:

    .. math::

        \text{PSNR} = 10 \log_{10} \bigg(\frac{\text{MAX}_I^2}{MSE(I,T)}\bigg)

    where

    .. math::

        \text{MSE}(I,T) = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1} [I(i,j) - T(i,j)]^2

    and :math:`\text{MAX}_I` is the maximum possible input value
    (e.g for floating point images :math:`\text{MAX}_I=1`).

    Args:
        input: the input image with arbitrary shape :math:`(*)`.
        labels: the labels image with arbitrary shape :math:`(*)`.
        max_val: The maximum value in the input tensor.

    Return:
        the computed loss as a scalar.

    Examples:
        >>> ones = torch.ones(1)
        >>> psnr(ones, 1.2 * ones, 2.) # 10 * log(4/((1.2-1)**2)) / log(10)
        tensor(20.0000)

    Reference:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    return 10.0 * torch.log10(max_val**2 / mse(input, target))