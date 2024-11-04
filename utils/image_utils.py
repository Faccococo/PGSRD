#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img

epsilon = 1e-6 
def get_gradient(image : torch.Tensor) -> torch.Tensor:
    
    sobel_x = torch.tensor([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    sobel_y = torch.tensor([[-1, -2, -1], 
                            [ 0,  0,  0], 
                            [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    
    image_copy = image.clone()
    
    grad_x = F.conv2d(image_copy, sobel_x, padding=1)
    grad_y = F.conv2d(image_copy, sobel_y, padding=1)
    
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)
    grad_magnitude = torch.sqrt(grad_magnitude)
    grad_magnitude = torch.sqrt(grad_magnitude)
    
    return grad_magnitude.squeeze(0)
