import pydiffvg
import torch.nn as nn
import torch
import skimage
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import string
import random
import torch.nn.functional as F

class primitive(nn.Module):
    ellipse = None
    canvas_width = None
    canvas_height = None
    type_lr = None
    color_lr = None
    rectangle_lr = None
    ellipse_lr = None
    triangle_lr = None
    render = pydiffvg.RenderFunction.apply
    print_values = True
    print_grads = True

    @classmethod
    def init_class_vars(cls, w, h, type_lr, color_lr, rectangle_lr, ellipse_lr, triangle_lr, order_lr, print_values, print_grads):
            
        cls.canvas_height = h
        cls.canvas_width = w
        cls.type_lr = type_lr
        cls.color_lr = color_lr
        cls.rectangle_lr = rectangle_lr
        cls.ellipse_lr = ellipse_lr
        cls.triangle_lr = triangle_lr
        cls.order_lr = order_lr
        
        cls.ellipse = pydiffvg.Ellipse(radius = torch.tensor([w * 0.5, h * 0.5]), center = torch.tensor([0.0, 0.0]))
        
        cls.print_values = print_values
        cls.print_grads = print_grads
        
        cls.f2b_mat = torch.tensor([[1.0/np.sqrt(3), 1.0],
                            [1.0/np.sqrt(3), -1.0],
                            [-2.0/np.sqrt(3), 0.0]]).float()
    
    def __init__(self,
                obb = np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                color_init = [0.5, 0.5, 0.5],
                type_select = [0.0, 0.0],
                depth = 0.0):
        super(primitive, self).__init__()
        
        if primitive.ellipse is None \
            or primitive.canvas_width is None \
            or primitive.canvas_height is None \
            or primitive.type_lr is None \
            or primitive.color_lr is None \
            or primitive.rectangle_lr is None \
            or primitive.ellipse_lr is None \
            or primitive.triangle_lr is None \
            or primitive.order_lr is None:
                raise Exception("Primitive class should be initialized.")
        
        self.age = 0
        self.depth = torch.tensor(depth, requires_grad=True)
        self.type_select = torch.tensor(type_select, requires_grad=True)
        assert self.type_select.shape == (2,)
        self.color = torch.tensor(color_init, requires_grad=True)
        assert self.color.shape == (3,)
        
        # specific settings for the rectangle
        # saving three points only
        self.rectangle_points = torch.tensor(obb[0:3], requires_grad=True)
        # could /np.array([primitive.canvas_width, primitive.canvas_height]) here
        
        # specific settings for the ellipse
        edge0 = obb[1, :] - obb[0, :]
        edge1 = obb[2, :] - obb[1, :]
        center = (obb[0, :] + obb[2, :]) / 2
        trans_mat = np.hstack([edge0.reshape(2, 1), edge1.reshape(2, 1)]) @ np.array([[1/primitive.canvas_width, 0.0], [0.0, -1/primitive.canvas_height]])
        
        translate = np.array([center[0]/primitive.canvas_width, center[1]/primitive.canvas_height]).reshape(2, 1)
        affine = np.hstack([trans_mat, translate])
        
        # before using the affine, we need to multiply the scaling
        self.ellipse_affine = torch.tensor(affine, requires_grad=True)
        
        # specific settings for the triangle
        triangle_pt0 = obb[0]
        triangle_pt1 = 0.5*(obb[1]+obb[2])
        triangle_pt2 = 0.5*(obb[2]+obb[3])
        triangle_pts = np.vstack([triangle_pt0, triangle_pt1, triangle_pt2])
        self.triangle_points = torch.tensor(triangle_pts, requires_grad=True)
        # could /np.array([primitive.canvas_width, primitive.canvas_height]) here
        
        self.optimizer_type = torch.optim.Adam([self.type_select], lr=primitive.type_lr)
        self.optimizer_color = torch.optim.Adam([self.color], lr=primitive.color_lr)
        self.optimizer_rectangle = torch.optim.Adam([self.rectangle_points], lr=primitive.rectangle_lr)
        self.optimizer_ellipse = torch.optim.Adam([self.ellipse_affine], lr=primitive.ellipse_lr)
        self.optimizer_triangle = torch.optim.Adam([self.triangle_points], lr=primitive.triangle_lr)
        original_color = torch.cat((self.color, torch.tensor([1.0])))
        self.optimizer_depth = torch.optim.Adam([self.depth], lr=primitive.order_lr)
        
        self.shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                fill_color = original_color,
                                shape_to_canvas = torch.eye(3))

    def forward(self):
        # Forward pass: render the image with three types
        self.type_dist = primitive.floats_to_blending(self.type_select)
        
        self.shape_group.fill_color = torch.cat((self.color, torch.tensor([1.0])))
        if primitive.print_values:
            # Print the current params.
            print('     color:', self.color)
            print('     type_select:', self.type_select)
            print('     type_prob:', self.type_dist)
        
        imgs = []
        
        # for rectangle
        imgs.append(self.render_rectangle())
        
        # for ellipse
        imgs.append(self.render_ellipse())
        
        # for triangle
        imgs.append(self.render_triangle())

        # you can define how to get this output img as you want
        img = imgs[0] * self.type_dist[0] + imgs[1] * self.type_dist[1] + imgs[2] * self.type_dist[2]
        
        return img
    
    def zero_grad(self):
        self.optimizer_color.zero_grad()
        self.optimizer_type.zero_grad()
        self.optimizer_rectangle.zero_grad()
        self.optimizer_ellipse.zero_grad()
        self.optimizer_triangle.zero_grad()
        self.optimizer_depth.zero_grad()
        
    def step(self):
        # Get avoid of being frozen due to very tiny p
        self.rectangle_points.grad = self.rectangle_points.grad / self.type_dist[0]
        self.ellipse_affine.grad = self.ellipse_affine.grad / self.type_dist[1]
        self.triangle_points.grad = self.triangle_points.grad / self.type_dist[2]
        
        if primitive.print_grads:
            # Print the gradients
            print('     type_select.grad:', self.type_select.grad)
        
        self.optimizer_color.step()
        self.optimizer_type.step()
        self.optimizer_rectangle.step()
        self.optimizer_ellipse.step()
        self.optimizer_triangle.step()
        self.optimizer_depth.step()
        self.age += 1
    
    def render_rectangle(self):
        self.shape_group.shape_to_canvas = torch.eye(3)
        new_pt = (self.rectangle_points[0]+self.rectangle_points[2]-self.rectangle_points[1]).reshape(1, 2)
        rec_points = torch.cat([self.rectangle_points, new_pt], dim=0)
        # rec_points[:, 0] = rec_points[:, 0] * primitive.canvas_width
        # rec_points[:, 1] = rec_points[:, 1] * primitive.canvas_height
        rectangle = pydiffvg.Polygon(points = rec_points, is_closed = True)
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            primitive.canvas_width, primitive.canvas_height, [rectangle], [self.shape_group])
        return primitive.render(primitive.canvas_width,   # width
                    primitive.canvas_height,   # height
                    2,     # num_samples_x
                    2,     # num_samples_y
                    0,   # seed
                    None, # background_image
                    *scene_args)
        
    def render_ellipse(self):
        self.shape_group.shape_to_canvas = torch.eye(3)
        affine = torch.cat((self.ellipse_affine, torch.tensor([[0.0, 0.0, 1.0]])), axis=0)
        affine[0, 2] = affine[0, 2] * primitive.canvas_width
        affine[1, 2] = affine[1, 2] * primitive.canvas_height
        # a very strange bug here
        affine_ = torch.eye(3)
        affine_[0, 2] = affine[0, 2]
        affine_[1, 2] = affine[1, 2]
        affine_[0, 0] = affine[0, 0]
        affine_[1, 1] = affine[1, 1]
        affine_[0, 1] = affine[0, 1]
        affine_[1, 0] = affine[1, 0]
        self.shape_group.shape_to_canvas = affine_
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            primitive.canvas_width, primitive.canvas_height, [primitive.ellipse], [self.shape_group])
        return primitive.render(primitive.canvas_width,   # width
                    primitive.canvas_height,   # height
                    2,     # num_samples_x
                    2,     # num_samples_y
                    0,   # seed
                    None, # background_image
                    *scene_args)
        
    def __str__(self):
        center_tri = (self.triangle_points[0] + self.triangle_points[1] + self.triangle_points[2]) / 3
        center_rec = (self.rectangle_points[0] + self.rectangle_points[2]) / 2
        center_elli = (self.ellipse_affine[0:2, 2]) * torch.tensor([primitive.canvas_width, primitive.canvas_height])
        
        return "Primitive: " + str(self.color) + " type: " + str(self.type_dist) + " rcenter: " + str(center_rec) + " ecenter: " + str(center_elli) + " tcenter: " + str(center_tri) + " depth: " + str(self.depth)
        
    def render_triangle(self):
        self.shape_group.shape_to_canvas = torch.eye(3)
        tri_points = self.triangle_points
        # tri_points = torch.zeros_like(self.triangle_points)
        # tri_points[:, 0] = self.triangle_points[:, 0] * primitive.canvas_width
        # tri_points[:, 1] = self.triangle_points[:, 1] * primitive.canvas_height
        triangle = pydiffvg.Polygon(points = tri_points, is_closed = True)
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            primitive.canvas_width, primitive.canvas_height, [triangle], [self.shape_group])
        return primitive.render(primitive.canvas_width,   # width
                    primitive.canvas_height,   # height
                    2,     # num_samples_x
                    2,     # num_samples_y
                    0,   # seed
                    None, # background_image
                    *scene_args)
        
    def render_MLE(self):
        type_prob = primitive.floats_to_blending(self.type_select)
        best_type = type_prob.tolist().index(max(type_prob))
        if best_type == 0:
            return self.render_rectangle()
        elif best_type == 1:
            return self.render_ellipse()
        else:
            return self.render_triangle()
        
    def from_instance(self):
                
        new_instance = primitive(color_init = self.color.tolist(), type_select = self.type_select.tolist(), depth = self.depth.item())
        new_instance.rectangle_points = self.rectangle_points.clone().detach().requires_grad_(True)
        new_instance.ellipse_affine = self.ellipse_affine.clone().detach().requires_grad_(True)
        new_instance.triangle_points = self.triangle_points.clone().detach().requires_grad_(True)
        new_instance.optimizer_rectangle = torch.optim.Adam([new_instance.rectangle_points], lr=primitive.rectangle_lr)
        new_instance.optimizer_ellipse = torch.optim.Adam([new_instance.ellipse_affine], lr=primitive.ellipse_lr)
        new_instance.optimizer_triangle = torch.optim.Adam([new_instance.triangle_points], lr=primitive.triangle_lr)
        new_instance.shape_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),fill_color = torch.cat((new_instance.color, torch.tensor([1.0]))),shape_to_canvas = torch.eye(3))
        
        return new_instance
    
    @classmethod
    def copy_rectangle(cls, instance, strength):
        if not isinstance(instance, cls) or isinstance(instance, background):
            raise TypeError(f"Expected an instance of {cls.__name__}, got {type(instance).__name__}")
        
        # compute obb
        points = instance.rectangle_points.detach().float().numpy()
        new_pt = (points[0]+points[2]-points[1]).reshape(1, 2)
        rec_points = np.concatenate([points, new_pt])
        rect = cv2.minAreaRect(rec_points)
        box = cv2.boxPoints(rect)
        
        # shrink box
        center = 0.5*(box[0] + box[2])
        box[0] = 0.5*(box[0] + center)
        box[1] = 0.5*(box[1] + center)
        box[2] = 0.5*(box[2] + center)
        box[3] = 0.5*(box[3] + center)
        
        # set type select:
        type_select = [1.0, np.sqrt(3)]
        type_select[0] *= strength
        type_select[1] *= strength
        
        new_instance = cls(color_init = instance.color.tolist(), obb=box, depth=instance.depth.item(), type_select=type_select)
        new_instance.type_dist = primitive.floats_to_blending(new_instance.type_select)
        
        new_instance.rectangle_points = instance.rectangle_points.clone().detach().requires_grad_(True)
        new_instance.optimizer_triangle = torch.optim.Adam([new_instance.rectangle_points], lr=primitive.rectangle_lr)

        
        return new_instance
    
    @classmethod
    def copy_ellipse(cls, instance, strength):
        if not isinstance(instance, cls) or isinstance(instance, background):
            raise TypeError(f"Expected an instance of {cls.__name__}, got {type(instance).__name__}")
        
        # set type_select
        type_select = [1.0, -np.sqrt(3)]
        type_select[0] *= strength
        type_select[1] *= strength
        
        # no obb is needed
        new_instance = cls(color_init = instance.color.tolist(), depth=instance.depth.item(), type_select= type_select)
        new_instance.ellipse_affine = instance.ellipse_affine.clone().detach().requires_grad_(True)
        
        # use this affine to compute the parallelogram
        affine = instance.ellipse_affine.detach().numpy()
        center = (affine[0:2, 2] * np.array([primitive.canvas_width, primitive.canvas_height])).reshape(-1)
        points = np.array([[-0.5*primitive.canvas_width, 0.5*primitive.canvas_height], [0.5*primitive.canvas_width, 0.5*primitive.canvas_height], [0.5*primitive.canvas_width, -0.5*primitive.canvas_height]]) @ affine[0:2, 0:2].T

        points = points + center
        # shrink points
        center = 0.5*(points[0] + points[2])
        points[0] = 0.5*(points[0] + center)
        points[1] = 0.5*(points[1] + center)
        points[2] = 0.5*(points[2] + center)
        points = points.astype(np.float32)
        new_instance.rectangle_points = torch.tensor(points[0:3], requires_grad=True)
        points = np.vstack([points, points[0]+points[2]-points[1]])
        points[1] = 0.5*(points[1]+points[2])
        points[2] = 0.5*(points[2]+points[3])
        new_instance.triangle_points = torch.tensor(points[0:3], requires_grad=True)
        
        new_instance.optimizer_rectangle = torch.optim.Adam([new_instance.rectangle_points], lr=primitive.rectangle_lr)
        new_instance.optimizer_ellipse = torch.optim.Adam([new_instance.ellipse_affine], lr=primitive.ellipse_lr)
        new_instance.optimizer_triangle = torch.optim.Adam([new_instance.triangle_points], lr=primitive.triangle_lr)
        
        new_instance.type_dist = primitive.floats_to_blending(new_instance.type_select)
        
        return new_instance
    
    @classmethod
    def copy_triangle(cls, instance, strength):
        if not isinstance(instance, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, got {type(instance).__name__}")
        
        # compute obb
        points = instance.triangle_points.detach().float().numpy()
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        
        # shrink box
        center = 0.5*(box[0] + box[2])
        box[0] = 0.5*(box[0] + center)
        box[1] = 0.5*(box[1] + center)
        box[2] = 0.5*(box[2] + center)
        box[3] = 0.5*(box[3] + center)
        
        # set type select
        type_select = [-2.0, 0.0]
        type_select[0] *= strength
        
        new_instance = cls(color_init = instance.color.tolist(), obb=box, depth=instance.depth.item(), type_select=type_select)
        
        new_instance.triangle_points = instance.triangle_points.clone().detach().requires_grad_(True)
        new_instance.optimizer_triangle = torch.optim.Adam([new_instance.triangle_points], lr=primitive.triangle_lr)
        
        new_instance.type_dist = primitive.floats_to_blending(new_instance.type_select)
        
        return new_instance
    
    @classmethod
    
    # the larger the split init is, the more solid in the type selection this new primitive is
    def split_shapes(cls, instance, threshold = 0.1, split_strength = 1):
        if not isinstance(instance, cls) or isinstance(instance, background):
            raise TypeError(f"Expected an instance of {cls.__name__}, got {type(instance).__name__}")
        
        ret = []
        
        if instance.type_dist[0] > threshold:
            new1 = primitive.copy_rectangle(instance, instance.type_dist[0]*split_strength)
            ret += [new1]
            
        if instance.type_dist[1] > threshold:
            new2 = primitive.copy_ellipse(instance, instance.type_dist[1]*split_strength)
            ret += [new2]
            
        if instance.type_dist[2] > threshold:
            new3 = primitive.copy_triangle(instance, instance.type_dist[2]*split_strength)
            ret += [new3]
        
        return ret
        
    @classmethod
    def floats_to_blending(cls, pt):
        xyz = cls.f2b_mat @ pt.unsqueeze(-1)
        xyz = xyz.squeeze()
        xyz = F.softmax(xyz)
        return xyz
    
    # the following functions are used for the annealing algorithm
    def set_annealing(self):
        self.old_annealing_type = np.random.randint(0, 3)
        
    def try_new_type_annealing(self):
        self.new_annealing_type = np.random.randint(0, 3)
        if self.new_annealing_type == 0:
            return self.render_rectangle()
        elif self.new_annealing_type == 1:
            return self.render_ellipse()
        else:
            return self.render_triangle()
    
    def cover_old_type_annealing(self):
        self.old_annealing_type = self.new_annealing_type
    
class background(primitive):
    def __init__(self, color_init = [0.5, 0.5, 0.5], change_color = False, depth = 0.0):
        super(background, self).__init__(color_init = color_init, obb = np.array([[-1.0, -1.0], [primitive.canvas_width+1.0, -1.0], [primitive.canvas_width+1.0, primitive.canvas_height+1.0], [-1.0, primitive.canvas_height+1.0]]), depth=depth)
        
        self.type_select = None
        self.optimizer_type = None
        self.type_dist = None
        self.triangle_points = None
        self.optimizer_triangle = None
        self.ellipse_affine = None
        self.optimizer_ellipse = None
        self.optimizer_rectangle = None
        self.rectangle_points = self.rectangle_points.requires_grad_(False)
        self.rectangle_points = self.rectangle_points.float()
        
        self.change_color = change_color
        
    def forward(self):
        
        self.shape_group.fill_color = torch.cat((self.color, torch.tensor([1.0])))
        if primitive.print_values:
            # Print the current params.
            print('     color:', self.color)
            
        return self.render_rectangle()
    
    def step(self):
        self.optimizer_depth.step()
        if self.change_color:
            self.optimizer_color.step()
            
    def zero_grad(self):
        self.optimizer_depth.zero_grad()
        if self.change_color:
            self.optimizer_color.zero_grad()
            
    def render_ellipse(self):
        print("Background should not render ellipse.")
        return
    
    def render_triangle(self):
        print("Background should not render triangle.")
        return
    
    def render_MLE(self):
        return self.render_rectangle()
    
    def split_shapes(self):
        print("Background should not split shapes.")
        return
    
    def from_instance(self):
        new_instance = background(color_init = self.color.tolist(), change_color = self.change_color)
        return new_instance
    
    def __str__(self):
        return "Background: " + str(self.color) + " depth: " + str(self.depth) + " points: " + str(self.rectangle_points)

def render_shapes_mixture(primitive_list, MLE_output = False, depth_reorder = False):
    img = torch.zeros((primitive.canvas_height, primitive.canvas_width, 3))
        
    ordered_list = primitive_list
    if depth_reorder == True:
        ordered_list = sorted(primitive_list, key=lambda x: x.depth.item(), reverse=True)
        
    # img has only three channels here
    for prim in ordered_list:
        # the deepest one is at the first
        if MLE_output:
            tmp_img = prim.render_MLE()
        else:
            tmp_img = prim.forward()
        # alpha blending
        img = img * (1-tmp_img[:,:,-1].unsqueeze(-1)) + tmp_img[:,:,0:3] * tmp_img[:,:,-1].unsqueeze(-1)
        
    return img

def alpha_blending_perpixel(colors, samples, para_bg = None):
    weight = 1
    assert(len(colors) == len(samples))
    colors_samples = list(zip(colors, samples))
    colors_samples = sorted(colors_samples, key=lambda x: x[1], reverse=False)
    # the greater the sample is, the deeper the color is.
    assert(len(colors) >= 0)
    total = colors_samples[0][0][0:3] * colors_samples[0][0][3] # total.shape = [3]
    weight *= 1 - colors_samples[0][0][3]
    # from top to bottom
    for (color, _) in colors_samples:
        if weight == 0: break
        total += color[0:3] * color[3] * weight
        weight *= 1 - color[3]
    
    return torch.tensor(total)

def render_shapes_reinforced_ddepth(primitive_list, gt_pyramid, pyramid_weights, number_loss_func, MLE_output = False, skip_portion = 20, layer_samples = 1, num_reg_samples = 100):
    
    h, w = primitive.canvas_height, primitive.canvas_width
    img = torch.zeros((h, w, 3))
    
    img_stack = []
    img_stack_alpha = torch.zeros((h, w))
    number_primitives = len(primitive_list)
    dorder = torch.zeros((number_primitives))
    depths = torch.tensor([p.depth for p in primitive_list])
    noskip = np.random.randint(0, skip_portion)
    
    pyramid_weights = pyramid_weights.reshape(-1, 1)
    
    # depth from larger to smaller, the deepest one should on the top
    # the original order in primitive_list keeps the same
    ordered_list = sorted(primitive_list, key=lambda x: x.depth.item(), reverse=True)
    indices_backward = [primitive_list.index(element) for element in ordered_list]
    indices_forward = [ordered_list.index(element) for element in primitive_list]
    # img has only three channels here
    for prim in ordered_list:
        # the deepest one is at the first
        if MLE_output:
            tmp_img = prim.render_MLE()
        else:
            tmp_img = prim.forward()
        # alpha blending
        img = img * (1-tmp_img[:,:,-1].unsqueeze(-1)) + tmp_img[:,:,0:3] * tmp_img[:,:,-1].unsqueeze(-1)
        img_stack.append(tmp_img)
        img_stack_alpha += tmp_img[:, :, -1]
    
    # reorder the image stack so that its indices can match the primitive_list
    img_stack_back = [img_stack[indices_forward[i]] for i in range(number_primitives)]
    
    with torch.no_grad():
        # compute the derivative for order manually
        for y in range(0, h):
            for x in range(0, w):
                if (y+x) % skip_portion != noskip: continue
                weight = img_stack_alpha[y, x]
                if weight == 0: continue
                
                colors = []
                centers = []
                indices = []
                
                for idx, layer in enumerate(img_stack_back):
                    if layer[y, x, 3] == 0: continue
                    colors.append(layer[y, x, :])
                    centers.append(primitive_list[idx].depth.item())
                    # here the indices will be mapped back to primitive list
                    indices.append(idx)
                
                if len(indices) <= 1: continue
                centers = torch.tensor(centers)
                tmp_dorder = None
                for _ in range(layer_samples):
                    rand = torch.randn(centers.shape)
                    samples = rand + centers
                    dp_p = rand # (dp/dcenter) / p
                    pixel_color = alpha_blending_perpixel(colors, samples).unsqueeze(0)
                    pixel_loss = torch.sum((pixel_color - gt_pyramid[:, y, x, :])**2 * pyramid_weights)
                    if tmp_dorder == None:
                        tmp_dorder = weight * pixel_loss * dp_p * skip_portion / layer_samples
                    else:
                        tmp_dorder += weight * pixel_loss * dp_p * skip_portion / layer_samples
                        
                assert tmp_dorder!=None
                for idx, l in enumerate(indices):
                        dorder[l] += tmp_dorder[idx]
                        
        # penalize the effective number of primitives
        # also reinforced derivative
        
        for _ in range(num_reg_samples):
            rand = torch.randn(number_primitives)
            samples = rand + depths
            count = torch.sum(samples >= samples[0])
            # count ranges from 1 to n, the larger number will cause smaller loss
            dp_p = rand # (dp/dcenter) / p
            dorder += number_loss_func(count) * dp_p / num_reg_samples
            
        for idx, prim in enumerate(primitive_list):
            prim.depth.grad = dorder[idx].detach()
        
    return img

def render_shapes_depth(primitive_list):
    
    h, w = primitive.canvas_height, primitive.canvas_width
    img = torch.zeros((h, w, 3))
    
    img_stack_weights = torch.zeros((h, w))
    
    for prim in primitive_list:
        tmp_img = prim.forward()
        img += tmp_img[:,:,0:3] * tmp_img[:,:,3:] * torch.exp(-prim.depth)
        img_stack_weights += tmp_img[:, :, -1] * torch.exp(-prim.depth)
        
    return img / img_stack_weights.unsqueeze(-1)
        
    