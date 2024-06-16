import os
import os.path as osp
import argparse
from easydict import EasyDict as edict
import torch
import datetime
import kornia
import torch.nn as nn

def get_experiment_id():
    current_time = datetime.datetime.now()
    ddhhss_string = current_time.strftime("%d%H%M%S")
    return ddhhss_string

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append(edict_2_dict(x[i]))
        return xnew
    else:
        return x

def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument('--log_dir', default="log")
    cfg = edict()
    args = parser.parse_args()
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    return cfg

def ycrcb_conversion(im, format='[bs x 3 x 2D]', reverse=False):
    mat = torch.FloatTensor([
        [ 65.481/255, 128.553/255,  24.966/255], # ranged_from [0, 219/255]
        [-37.797/255, -74.203/255, 112.000/255], # ranged_from [-112/255, 112/255]
        [112.000/255, -93.786/255, -18.214/255], # ranged_from [-112/255, 112/255]
    ]).to(im.device)

    if reverse:
        mat = mat.inverse()

    if format == '[bs x 3 x 2D]':
        im = im.permute(0, 2, 3, 1)
        im = torch.matmul(im, mat.T)
        im = im.permute(0, 3, 1, 2).contiguous()
        return im
    elif format == '[2D x 3]':
        im = torch.matmul(im, mat.T)
        return im
    else:
        raise ValueError
    
def pyramid_loss(img, gt, pyramid_levels):
    
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    
    assert img.shape == gt.shape, "Input and target images must have the same shape"
    
    # (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
    img = img.permute(2, 0, 1).unsqueeze(0)
    gt = gt.permute(2, 0, 1).unsqueeze(0) 
    
    for level in pyramid_levels:
        kernel_size = (2 * level + 1, 2 * level + 1)
        sigma = (level, level)
        
        blurred_img = kornia.filters.gaussian_blur2d(img, kernel_size, sigma)
        
        loss = mse_loss(blurred_img, gt)
        
        total_loss += loss
    
    return total_loss
