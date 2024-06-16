"""
Here is the use case:
python compare.py --config config/girl.yaml --target imgs/portrait.png
"""


import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
import math
import errno
from tqdm import tqdm
from primitive import primitive, render_shapes_mixture, render_shapes_reinforced_ddepth, background, render_shapes_depth
from initialization import initialize, CycleInDAGError
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import PIL
import PIL.Image
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import shutil
import copy

import yaml
from easydict import EasyDict as edict
from utils import *

if __name__ == "__main__":

    ###############
    # make config #
    ###############

    cfg_arg = parse_args()
    with open(cfg_arg.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(cfg)
    cfg.update(cfg_arg)
    cfg.exid = get_experiment_id()

    cfg.experiment_dir = osp.join(cfg.log_dir, f'{cfg.exid}')
    configfile = osp.join(cfg.experiment_dir, 'config.yaml')
    check_and_create_dir(configfile)
    with open(osp.join(configfile), 'w') as f:
        yaml.dump(edict_2_dict(cfg), f)
    #print(cfg)
        
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()
    
    gt = np.array(PIL.Image.open(cfg.target))
    print(f"Input image shape is: {gt.shape}")
    if len(gt.shape) == 2:
        print("Converting the gray-scale image to RGB.")
        gt = gt.unsqueeze(dim=-1).repeat(1,1,3)
    if gt.shape[2] == 4:
        print("Input image includes alpha channel, simply dropout alpha channel.")
        gt = gt[:, :, :3]
    gt = (gt/255).astype(np.float32)
    gt = torch.FloatTensor(gt).to(device)
    h, w = gt.shape[:-1]
    # gt is (h, w, channels), here the channel number is 3
    

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
    primitive_list = initialize(gt, cfg, convex_info=False)
    
    n = len(primitive_list)
    
    # copy the initialization
    primitive_list1 = [prim.from_instance() for prim in primitive_list]
    
    # Bulid pyramid for gt
    gt_pyramid = [gt]
    k_size = ((int)(0.25*h)*2 + 1, (int)(0.25*w)*2 + 1)
    pyramid_base = gt.numpy()
    for i in range(cfg.pyramid_levels - 1):
        new_level = cv2.GaussianBlur(pyramid_base, ksize=k_size, sigmaX=2.0**(i+1))
        gt_pyramid.append(torch.FloatTensor(new_level).to(device))
        
    gt_pyramid = torch.stack(gt_pyramid, dim=0)
    # Size = [level, H, W, 3]
    weights = torch.tensor([0.5**i for i in range(cfg.pyramid_levels)])
    weights = weights.view(-1, 1, 1, 1)
    
    # for image without background, the deletion is not well defined!
    assert isinstance(primitive_list[0], background)
        
    ####################
    # start training 1 #
    ####################
    
    primitive_list = primitive_list[:cfg.kmeans.max_fragments]
    n = len(primitive_list)

    print(f"=> Optimizing {n} primitives...")

    if cfg.save.init:
        filename = os.path.join(cfg.experiment_dir, f"init{n}shapes_reorder.png")
        check_and_create_dir(filename)
        img = render_shapes_mixture(primitive_list, )
        pydiffvg.imwrite(img, filename, gamma=1.0)
        
    # Inner loop training
    t_range = tqdm(range(cfg.num_iter))
    
    loss_record = []
    number_record = []
    mle_record = []
    
    # define number regularizaton
    def number_loss_func(effective_number):
        return -effective_number*cfg.regularization.number
    
    # determine whether this primitive should be split
    def split_judge(primitive):
        if primitive.age < cfg.split.age_threshold:
            return False
        
        p0 = primitive.type_dist[0].item()
        p1 = primitive.type_dist[1].item()
        p2 = primitive.type_dist[2].item()
        
        d0 = (p0-1)**2 + p1**2 + p2**2
        d1 = p0**2 + (p1-1)**2 + p2**2
        d2 = p0**2 + p1**2 + (p2-1)**2
        d = min(d0, d1, d2)
        d = np.sqrt(d)
        
        if d > cfg.split.split_threshold / np.log(primitive.age):
            return True
        else:
            return False
        
    delete_threshold = torch.distributions.Normal(0, 1).icdf(torch.tensor(cfg.split.delete_threshold)).item()
    
    for t in t_range:
        
        # Delete hidden layers
        # If there is no background, how do we delete the layers?
        primitive_list = [prim for prim in primitive_list if prim.depth - primitive_list[0].depth <= delete_threshold]
        
        n = len(primitive_list)
                
        # Split ambiguous layers:
        for i in range(1, n):
            if split_judge(primitive_list[i]):
                tmp = primitive.split_shapes(primitive_list[i], cfg.split.reserve_threshold)
                primitive_list.pop(i)
                primitive_list += tmp
        
        n = len(primitive_list)
        number_record.append(n)
        
        for p in primitive_list:
            p.zero_grad()
            
        # Blurring the gt instead of blurring the rendered image
        img = render_shapes_reinforced_ddepth(primitive_list, gt_pyramid, weights, number_loss_func, skip_portion=cfg.skip_portion, layer_samples=cfg.layer_samples, num_reg_samples=cfg.penal_samples)
        pydiffvg.imwrite(img.cpu(), cfg.experiment_dir+f'/iter_{t}_ours.png', gamma=1.0)
        
        # Compute the loss function. Here it is L2.
        img = img.unsqueeze(0)
        diffs = img - gt_pyramid
        squared_diffs = diffs.pow(2)
        weighted_squared_diffs = squared_diffs * weights
        loss = weighted_squared_diffs.sum()
        
        # Regularization for type selection
        for p in primitive_list[1:]:
            loss += cfg.regularization.type * (p.type_dist[0]*p.type_dist[1]+p.type_dist[1]*p.type_dist[2]+p.type_dist[2]*p.type_dist[0])
            
        # Regularization for depth
        depth_avg = 0.0
        for p in primitive_list:
            depth_avg += p.depth / n
        
        for p in primitive_list:
            loss += cfg.regularization.depth * (p.depth - depth_avg)**2
        
        if cfg.print_values:
            print('loss:', loss.item())

        loss.backward()
        
        for p in primitive_list:
            p.step()
        
        if cfg.save.loss:
            loss_record.append(weighted_squared_diffs.sum().item())
            
        if cfg.save.mle:
            with torch.no_grad():
                MLE_img = render_shapes_mixture(primitive_list, MLE_output=True, depth_reorder=True)
                pydiffvg.imwrite(MLE_img.cpu(), cfg.experiment_dir+f'/iter_{t}_mle_ours.png', gamma=1.0)
                MLE_img = MLE_img.unsqueeze(0)
                diffs = MLE_img - gt_pyramid
                squared_diffs = diffs.pow(2)
                weighted_squared_diffs = squared_diffs * weights
                loss = weighted_squared_diffs.sum()
                mle_record.append(loss.item())
    
    # remove useless layers
    primitive_list = [prim for prim in primitive_list if prim.depth - primitive_list[0].depth <= 0.0]
    n = len(primitive_list)
    number_record.append(n)
            
    ##################
    #  save outcomes #
    ##################

    if cfg.save.output:
        img = render_shapes_mixture(primitive_list, MLE_output=True, depth_reorder=True)
        filename = os.path.join(cfg.experiment_dir, f"output_ours.png")
        pydiffvg.imwrite(img, filename, gamma=1.0)
        txtname = os.path.join(cfg.experiment_dir, f"output_ours.txt")
        with open(txtname, 'w') as file:
            for prim in primitive_list:
                file.write(str(prim))
                
        # Compute the final loss function.
        img.unsqueeze(0)
        diffs = img - gt_pyramid
        squared_diffs = diffs.pow(2)
        weighted_squared_diffs = squared_diffs * weights
        final_loss = weighted_squared_diffs.sum()

    if cfg.save.visualize_optimization:
        from subprocess import call
        call(["ffmpeg", "-framerate", "24", "-i",
            cfg.experiment_dir+"/iter_%d_ours.png", "-vb", "20M",
            cfg.experiment_dir+"/out_ours.mp4"])
        if cfg.save.mle:
            call(["ffmpeg", "-framerate", "24", "-i",
                cfg.experiment_dir+"/iter_%d_mle_ours.png", "-vb", "20M",
                cfg.experiment_dir+"/out_ours_mle.mp4"])
        
    if cfg.save.loss:
        plt.clf()
        plt.plot(range(cfg.num_iter), loss_record, linestyle='-')
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_ours.png")
        plt.savefig(filename)
        
        if cfg.save.mle:
            plt.clf()
            plt.plot(range(cfg.num_iter), mle_record, linestyle='-')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('MLE Loss')
            #plt.grid(True)
            filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_ours_mle.png")
            plt.savefig(filename)
        
        
        loss_txt_filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_ours.txt")
        with open(loss_txt_filename, 'w') as f:
            for i, loss in enumerate(loss_record):
                f.write(f"Iteration {i}: {loss}\n")
            
        if cfg.save.mle:
            loss_txt_filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_ours_mle.txt")
            with open(loss_txt_filename, 'w') as f:
                for i, loss in enumerate(mle_record):
                    f.write(f"Iteration {i}: {loss}\n")
                f.write(f"Final loss: {final_loss}\n")
        
        plt.clf()
        plt.plot(range(len(number_record)), number_record, linestyle='-')
        plt.title('Number of primitives over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Number of primitives')
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"number_for_{cfg.num_iter}iters_ours.png")
        plt.savefig(filename)
        
    ####################
    # start training 2 #
    ####################
    
    primitive_list1 = primitive_list1[:number_record[-1]]
    n = len(primitive_list1)
    print(f"=> Optimizing {n} primitives...")

    if cfg.save.init:
        filename = os.path.join(cfg.experiment_dir, f"init{n}shapes_reddy.png")
        check_and_create_dir(filename)
        img = render_shapes_mixture(primitive_list1, )
        pydiffvg.imwrite(img, filename, gamma=1.0)
        
    # Inner loop training
    t_range = tqdm(range(cfg.num_iter))
    
    loss_record1 = []
    number_record1 = []
    mle_record1 = []
    
    # # define number regularizaton
    # def number_loss_func(effective_number):
    #     return -effective_number*cfg.regularization.number
    
    # # determine whether this primitive should be split
    # def split_judge(primitive):
    #     if primitive.age < cfg.split.age_threshold:
    #         return False
        
    #     p0 = primitive.type_dist[0].item()
    #     p1 = primitive.type_dist[1].item()
    #     p2 = primitive.type_dist[2].item()
        
    #     d0 = (p0-1)**2 + p1**2 + p2**2
    #     d1 = p0**2 + (p1-1)**2 + p2**2
    #     d2 = p0**2 + p1**2 + (p2-1)**2
    #     d = min(d0, d1, d2)
    #     d = np.sqrt(d)
        
    #     if d > cfg.split.split_threshold / np.log(primitive.age):
    #         return True
    #     else:
    #         return False
    
    for t in t_range:
        
        # Delete hidden layers
        if t % cfg.reddy.delete_freq == 0:
            primitive_list1 = [prim for prim in primitive_list1 if prim.depth <= primitive_list1[0].depth]
        
        n = len(primitive_list1)
                
        # # Split ambiguous layers:
        # for i in range(1, n):
        #     if split_judge(primitive_list[i]):
        #         tmp = primitive.split_shapes(primitive_list[i], cfg.split.reserve_threshold)
        #         primitive_list.pop(i)
        #         primitive_list += tmp
        
        # n = len(primitive_list)
        number_record1.append(n)
        
        for p in primitive_list1:
            p.zero_grad()
            
        # Blurring the gt instead of blurring the rendered image
        img = render_shapes_depth(primitive_list1)
        pydiffvg.imwrite(img.cpu(), cfg.experiment_dir+f'/iter_{t}_reddy.png', gamma=1.0)
        
        # Compute the loss function. Here it is L2.
        img = img.unsqueeze(0)
        diffs = img - gt_pyramid
        squared_diffs = diffs.pow(2)
        weighted_squared_diffs = squared_diffs * weights
        loss = weighted_squared_diffs.sum()
        
        # # Regularization for type selection
        # for p in primitive_list[1:]:
        #     loss += cfg.regularization.type * (p.type_dist[0]*p.type_dist[1]+p.type_dist[1]*p.type_dist[2]+p.type_dist[2]*p.type_dist[0])
            
        # # Regularization for depth
        # depth_avg = 0.0
        # for p in primitive_list:
        #     depth_avg += p.depth / n
        
        # for p in primitive_list:
        #     loss += cfg.regularization.depth * (p.depth - depth_avg)**2
        
        if cfg.print_values:
            print('loss:', loss.item())

        loss.backward()
        
        for p in primitive_list1:
            p.step()
        
        if cfg.save.loss:
            loss_record1.append(weighted_squared_diffs.sum().item())
            
        if cfg.save.mle:
            with torch.no_grad():
                MLE_img = render_shapes_mixture(primitive_list1, MLE_output=True, depth_reorder=True)
                pydiffvg.imwrite(MLE_img.cpu(), cfg.experiment_dir+f'/iter_{t}_mle_reddy.png', gamma=1.0)
                MLE_img = MLE_img.unsqueeze(0)
                diffs = MLE_img - gt_pyramid
                squared_diffs = diffs.pow(2)
                weighted_squared_diffs = squared_diffs * weights
                mle_record1.append(weighted_squared_diffs.sum().item())
    
    # remove useless layers
    primitive_list1 = [prim for prim in primitive_list1 if prim.depth - primitive_list1[0].depth <= 0.0]
    n = len(primitive_list1)
    number_record1.append(n)
            
            
    ##################
    #  save outcomes #
    ##################

    if cfg.save.output:
        img = render_shapes_mixture(primitive_list1, MLE_output=True, depth_reorder=True)
        filename = os.path.join(cfg.experiment_dir, f"output_reddy.png")
        pydiffvg.imwrite(img, filename, gamma=1.0)
        txtname = os.path.join(cfg.experiment_dir, f"output_reddy.txt")
        with open(txtname, 'w') as file:
            for prim in primitive_list1:
                file.write(str(prim))
                
        # Compute the final loss function.
        img.unsqueeze(0)
        diffs = img - gt_pyramid
        squared_diffs = diffs.pow(2)
        weighted_squared_diffs = squared_diffs * weights
        final_loss = weighted_squared_diffs.sum()

    if cfg.save.visualize_optimization:
        from subprocess import call
        call(["ffmpeg", "-framerate", "24", "-i",
            cfg.experiment_dir+"/iter_%d_reddy.png", "-vb", "20M",
            cfg.experiment_dir+"/out_reddy.mp4"])
        if cfg.save.mle:
            call(["ffmpeg", "-framerate", "24", "-i",
                cfg.experiment_dir+"/iter_%d_mle_reddy.png", "-vb", "20M",
                cfg.experiment_dir+"/out_mle_reddy.mp4"])
        
    if cfg.save.loss:
        plt.clf()
        plt.plot(range(cfg.num_iter), loss_record1, linestyle='-')
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss (Reddy)')
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_reddy.png")
        plt.savefig(filename)
        
        if cfg.save.mle:
            plt.clf()
            plt.plot(range(cfg.num_iter), mle_record1, linestyle='-')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('MLE Loss (Reddy)')
            #plt.grid(True)
            filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_reddy_mle.png")
            plt.savefig(filename)
        
        
        loss_txt_filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_reddy.txt")
        with open(loss_txt_filename, 'w') as f:
            for i, loss in enumerate(loss_record1):
                f.write(f"Iteration {i}: {loss}\n")
            
        if cfg.save.mle:
            loss_txt_filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_reddy_mle.txt")
            with open(loss_txt_filename, 'w') as f:
                for i, loss in enumerate(mle_record1):
                    f.write(f"Iteration {i}: {loss}\n")
                f.write(f"Final loss: {final_loss}\n")
        
        plt.clf()
        plt.plot(range(len(number_record1)), number_record1, linestyle='-')
        plt.title('Number of primitives over iterations')
        plt.xlabel('Iterations (Reddy)')
        plt.ylabel('Number of primitives')
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"number_for_{cfg.num_iter}iters_reddy.png")
        plt.savefig(filename)
        
        # output comparison curves
        plt.clf()
        plt.plot(range(cfg.num_iter), loss_record, linestyle='-', label='ours')
        plt.plot(range(cfg.num_iter), loss_record1, linestyle='-', label='Reddy')
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_comparison.png")
        plt.savefig(filename)
        
        # output comparison MLE curves
        plt.clf()
        plt.plot(range(cfg.num_iter), mle_record, linestyle='-', label='ours')
        plt.plot(range(cfg.num_iter), mle_record1, linestyle='-', label='Reddy')
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('MLE Loss')
        plt.legend()
        #plt.grid(True)
        filename = os.path.join(cfg.experiment_dir, f"loss_for_{cfg.num_iter}iters_comparison_mle.png")
        plt.savefig(filename)