"""
Here is the use case:
python test.py --config config/test.yaml
"""

import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
import math
import errno
from tqdm import tqdm
from primitive import primitive, render_shapes_mixture, render_shapes_reinforced_ddepth
from initialization import initialize, CycleInDAGError
import warnings
warnings.filterwarnings("ignore")

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
    
    w = cfg.test.w
    h = cfg.test.h
    
    def random_point():
        x = float(np.random.uniform(0, w))
        np.random.seed(int(x*x+1145141*x))
        y = float(np.random.uniform(0, h))
        np.random.seed(int(x+y+x*y+1145141*y))
        return [x, y]

    def triangle_area(p1, p2, p3):
        def cross_product(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0]

        def subtract_vectors(v1, v2):
            return [v1[0] - v2[0], v1[1] - v2[1]]
        v1 = subtract_vectors(p2, p1)
        v2 = subtract_vectors(p3, p1)
        cross = cross_product(v1, v2)
        return abs(cross) / 2

    def is_center_point_inside_triangle(p1, p2, p3):
        def sign(x1, y1, x2, y2, x3, y3):
            return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

        b1 = sign(0.5*w, 0.5*h, p1[0], p1[1], p2[0], p2[1]) < 0.0
        b2 = sign(0.5*w, 0.5*h, p2[0], p2[1], p3[0], p3[1]) < 0.0
        b3 = sign(0.5*w, 0.5*h, p3[0], p3[1], p1[0], p1[1]) < 0.0

        return (b1 == b2) and (b2 == b3)
    
    for test_i in range(cfg.test.num):
        # randomly generate inputs
        print(f'=> Test number is {test_i}.....')
        
        shapes = []
        shape_groups = []
        for _ in range(cfg.test.shape_num):
            rnd = random.randint(0, 2)
            if rnd == 0:
                # generate triangle
                p1 = random_point()
                p2 = random_point()
                p3 = random_point()
                # make it easier to overlapping
                while is_center_point_inside_triangle(p1, p2, p3) or triangle_area(p1, p2, p3)<0.05*w*h:
                    p1 = random_point()
                    p2 = random_point()
                    p3 = random_point()
                triangle_points = torch.tensor([p1, p2, p3]).float()
                tmp_shape = pydiffvg.Polygon(points = triangle_points, is_closed = True)
                tmp_shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]), fill_color = torch.rand((4,)).float())
            elif rnd == 1:
                # generate parallelogram
                p1 = random_point()
                p2 = random_point()
                p3 = random_point()
                p4 = [p3[0]+p1[0]-p2[0], p3[1]+p1[1]-p2[1]]
                while is_center_point_inside_triangle(p1, p2, p3) or is_center_point_inside_triangle(p1, p3, p4) or triangle_area(p1, p2, p3)<0.05*w*h or p4[0]>w or p4[0]<0 or p4[1]>h or p4[1]<0:
                    p1 = random_point()
                    p2 = random_point()
                    p3 = random_point()
                    p4 = [p3[0]+p1[0]-p2[0], p3[1]+p1[1]-p2[1]]
                rectangle_points = torch.tensor([p1, p2, p3, p4]).float()
                tmp_shape = pydiffvg.Polygon(points = rectangle_points, is_closed = True)
                tmp_shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                fill_color = torch.rand((4,)).float())
            elif rnd == 2:
                # generate ellipse
                tmp_shape = pydiffvg.Ellipse(radius = torch.tensor([w * 0.3, h * 0.1]), center = torch.tensor([0.0, 0.0]))
                rotate = torch.zeros((2, 2))
                theta = torch.tensor(np.pi * random.uniform(0, 1))
                rotate[0, 0] = torch.cos(theta)
                rotate[0, 1] = -torch.sin(theta)
                rotate[1, 0] = torch.sin(theta)
                rotate[1, 1] = torch.cos(theta)
                
                p1 = random_point()
                p2 = random_point()
                center = [0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])]
                
                mat = torch.eye(3)
                mat[0:2, 0:2] = rotate
                mat[0, 2] = center[0]
                mat[1, 2] = center[1]
                    
                tmp_shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                fill_color = torch.rand((4,)).float(),
                                shape_to_canvas=mat)
            shapes.append(tmp_shape)
            shape_groups.append(tmp_shape_group)
        gt = torch.ones((h, w, 4)).float()
        render = pydiffvg.RenderFunction.apply
        for idx, shape in enumerate(shapes):
            shape_groups[idx].fill_color[3] = 1.0
            scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, [shape], [shape_groups[idx]])
            gt = render(w,   # width
                        h,   # height
                        2,     # num_samples_x
                        2,     # num_samples_y
                        0,   # seed
                        gt, # background_image
                        *scene_args)
        
        gt = gt[:,:,0:3]
        tmp_dir = osp.join(cfg.experiment_dir, f'{test_i}')
        pydiffvg.imwrite(gt, f'{tmp_dir}/input.png', gamma=1.0)
        
        bg_img = torch.ones((h, w, 4))
        
        ####################
        # start training 1 #
        ####################
        
        try:
            primitive_list = initialize(gt, cfg)
        except CycleInDAGError as e:
            continue
        n = len(primitive_list)

        print(f"=> Optimizing {n} primitives...")

        if cfg.save.init:
            filename = os.path.join(tmp_dir, f"init{n}shapes.png")
            check_and_create_dir(filename)
            img = render_shapes_mixture(primitive_list, bg_img)
            pydiffvg.imwrite(img, filename, gamma=1.0)
            
        # Inner loop training
        t_range = tqdm(range(cfg.num_iter))
        
        loss_record1 = []
        depth = torch.zeros((len(primitive_list)))
        # read depth out:
        for idx, p in enumerate(primitive_list):
            depth[idx] = p.depth
        depth_adam = torch.optim.Adam([depth], lr=cfg.lr.order)
        for t in t_range:
            for p in primitive_list:
                p.zero_grad()
            depth_adam.zero_grad()
            img, ddepth = render_shapes_reinforced_ddepth(primitive_list, gt, bg_img)
            pydiffvg.imwrite(img.cpu(), tmp_dir+f'/iter_{t}.png', gamma=1.0)
            
            # Compute the loss function. Here it is L2.
            loss = (img - gt).pow(2).sum()
            if cfg.print_values:
                print('loss:', loss.item())

            loss.backward()
            depth.grad = ddepth
            
            for p in primitive_list:
                p.step()
            depth_adam.step()
            
            
            # Update order to primitive_list:
            for idx, layer in enumerate(primitive_list):
                layer.depth = depth[idx]
            
            if cfg.save.loss:
                loss_record1.append(loss.item())
                
        ##################
        #  save outcomes #
        ##################

        if cfg.save.output:
            img = render_shapes_mixture(primitive_list, bg_img, MLE_output=True, depth_reorder=True)
            filename = os.path.join(tmp_dir, f"output.png")
            pydiffvg.imwrite(img, filename, gamma=1.0)

        if cfg.save.visualize_optimization:
            from subprocess import call
            call(["ffmpeg", "-framerate", "24", "-i",
                tmp_dir+"/iter_%d.png", "-vb", "20M",
                tmp_dir+"/out.mp4"])
            
        if cfg.save.loss:
            plt.clf()
            plt.plot(range(cfg.num_iter), loss_record1, linestyle='-')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            #plt.grid(True)
            filename = os.path.join(tmp_dir, f"loss_for_{cfg.num_iter}iters.png")
            plt.savefig(filename)
            
        ####################
        # start training 2 #
        ####################

        primitive_list = initialize(gt, cfg, False)
        n = len(primitive_list)

        print(f"=> Optimizing {n} primitives...")

        if cfg.save.init:
            filename = os.path.join(tmp_dir, f"init{n}shapes_naive.png")
            check_and_create_dir(filename)
            img = render_shapes_mixture(primitive_list, bg_img)
            pydiffvg.imwrite(img, filename, gamma=1.0)
            
        # Inner loop training
        t_range = tqdm(range(cfg.num_iter))
        
        loss_record2 = []
        depth = torch.zeros((len(primitive_list)))
        # read depth out:
        for idx, p in enumerate(primitive_list):
            depth[idx] = p.depth
        depth_adam = torch.optim.Adam([depth], lr=cfg.lr.order)
        for t in t_range:
            for p in primitive_list:
                p.zero_grad()
            depth_adam.zero_grad()
            img, ddepth = render_shapes_reinforced_ddepth(primitive_list, gt, bg_img)
            pydiffvg.imwrite(img.cpu(), tmp_dir+f'/iter_{t}_naive.png', gamma=1.0)
            
            # Compute the loss function. Here it is L2.
            loss = (img - gt).pow(2).sum()
            if cfg.print_values:
                print('loss:', loss.item())

            loss.backward()
            depth.grad = ddepth
            
            for p in primitive_list:
                p.step()
            depth_adam.step()
            
            
            # Update order to primitive_list:
            for idx, layer in enumerate(primitive_list):
                layer.depth = depth[idx]
            
            if cfg.save.loss:
                loss_record2.append(loss.item())
                
        ##################
        #  save outcomes #
        ##################

        if cfg.save.output:
            img = render_shapes_mixture(primitive_list, bg_img, MLE_output=True, depth_reorder=True)
            filename = os.path.join(tmp_dir, f"output_naive.png")
            pydiffvg.imwrite(img, filename, gamma=1.0)

        if cfg.save.visualize_optimization:
            from subprocess import call
            call(["ffmpeg", "-framerate", "24", "-i",
                tmp_dir+"/iter_%d_naive.png", "-vb", "20M",
                tmp_dir+"/out_naive.mp4"])
            
        if cfg.save.loss:
            plt.clf()
            plt.plot(range(cfg.num_iter), loss_record2, linestyle='-')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            #plt.grid(True)
            filename = os.path.join(tmp_dir, f"loss_for_{cfg.num_iter}iters_naive.png")
            plt.savefig(filename)
        
        if cfg.save.loss:
            plt.clf()
            plt.plot(range(cfg.num_iter), loss_record1, linestyle='-', color='blue', label='Loss')
            plt.plot(range(cfg.num_iter), loss_record2, linestyle='-', color='red', label='Naive Loss')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()  # 添加图例
            #plt.grid(True)
            filename = os.path.join(tmp_dir, f"loss_for_{cfg.num_iter}iters_compare.png")
            plt.savefig(filename)
        
    
        