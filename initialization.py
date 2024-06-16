import pydiffvg
import torch.nn as nn
import torch
import skimage
import numpy as np
import cv2
import math
from PIL import Image
import torchvision.transforms as transforms
import string
import random
import torch.nn.functional as F
from primitive import primitive, background
import networkx as nx
import matplotlib.pyplot as plt
import copy

class contour_record:
    boundary_error = 3
    def __init__(self, color, contours):
        if not isinstance(contours, list):
            contours = [contours]
        self.color = color #RGB, if use cv2 to output it should be BGR
        self.contours = contours
        # self.contours == [(poly1points, 1, 2), ..., (polynpoints, 1, 2)]
        self.update_area()
        self.update_pos()
        self.update_convex_hull()
        self.update_extra_area()
        self.lower_layers = set()
        
    def __str__(self):
        center_x = int(self.translation[0] + 0.5 * self.w)
        center_y = int(self.translation[1] + 0.5 * self.h)
        return f"{(center_x, center_y)}"
    
    def update_area(self):
        self.area = 0
        for ct in self.contours:
            self.area += cv2.contourArea(ct)
    
    def update_pos(self):
        # if take vstack, it will become a single polygon (allpolypoints, 1, 2)
        lu_x, lu_y, self.w, self.h = cv2.boundingRect(np.vstack(self.contours))
        self.translation = np.array([lu_x, lu_y])
    
    def update_convex_hull(self):
        # if take vstack, it will become a single polygon (allpolypoints, 1, 2)
        self.convex_hull = cv2.convexHull(np.vstack(self.contours))
        self.image = np.zeros((self.h, self.w), dtype=np.uint8)
        for single_block in self.contours:
            cv2.fillPoly(self.image, [single_block - self.translation], 255)
            # self.image expands several pixels
            cv2.polylines(self.image, [single_block - self.translation], isClosed=True, color=255, thickness=contour_record.boundary_error)
    
    def update_extra_area(self):
        self.extra_region = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(self.extra_region, [self.convex_hull - self.translation], 255)
        self.extra_region = cv2.bitwise_and(self.extra_region, 255 - self.image)
        # self.extra_area contracts several pixels
        cv2.polylines(self.extra_region, [self.convex_hull - self.translation], isClosed=True, color=0, thickness=contour_record.boundary_error)
        self.extra_area = cv2.countNonZero(self.extra_region)
        return self.extra_area
        
    # Only erase self.extra_region, ct_rec won't be changed but self will
    def get_erased_by(self, ct_rec):
        a = ct_rec.intersect_other_image(self.extra_region, self.translation, erase=True)
        #assert self.extra_area - a == cv2.countNonZero(self.extra_region)
        self.extra_area -= a
        return a
        
    # The return value is an int of the intersection area with self.image; If the return value is positive and the erase is True, the input image will be erased by the self.image
    # Warning: The original item.extra_area may be changed as well, the user needs to call item.update_extra_area()
    def intersect_other_image(self, input_img, input_translation, erase=False):
        # determine whether they intersect roughly using aabb
        x_min, y_min = self.translation
        x_max = x_min + self.w
        y_max = y_min + self.h
        input_x_min, input_y_min = input_translation
        input_x_max, input_y_max = input_translation + input_img.shape[::-1]
        
        if x_max < input_x_min or input_x_max < x_min or y_max < input_y_min or input_y_max < y_min:
            return 0
    
        x_offset = x_min - input_x_min
        y_offset = y_min - input_y_min
        input_start_x = max(0, x_offset)
        input_end_x = min(input_img.shape[1], x_offset + self.w)
        input_start_y = max(0, y_offset)
        input_end_y = min(input_img.shape[0], y_offset + self.h)
        self_start_x = max(0, -x_offset)
        self_end_x = min(self.w, -x_offset + input_img.shape[1])
        self_start_y = max(0, -y_offset)
        self_end_y = min(self.h, -y_offset + input_img.shape[0])
        
        intersection = np.zeros_like(input_img, dtype=np.uint8)
        intersection[input_start_y:input_end_y, input_start_x:input_end_x] = \
        self.image[self_start_y:self_end_y, self_start_x:self_end_x]
        intersection = cv2.bitwise_and(input_img, intersection)
        count = cv2.countNonZero(intersection)
        
        if erase:
            input_img[intersection == 255] = 0
        
        return count
    
    def output_primitive(self, depth=0.0):
        rect = cv2.minAreaRect(np.vstack(self.contours))
        box = cv2.boxPoints(rect)
        
        return primitive(box,
                         color_init=self.color,
                         depth=depth)
        
        
    def copy_from(self, other):
        # shallow copy
        self.__dict__.update(copy.copy(other.__dict__))
        
    def merge_from(self, other):
        assert (self.color == other.color).all()
        self.contours += other.contours
        self.lower_layers |= other.lower_layers
        self.update_area()
        self.update_pos()
        self.update_convex_hull()
        self.update_extra_area()
        
    def include(self, other):
        a = self.intersect_other_image(other.image, other.translation)
        if a == cv2.countNonZero(other.image):
            return True
        else:
            return False
            
    
def generate_new_contour(ct1, ct2):
    assert (ct1.color == ct2.color).all()
    return contour_record(ct1.color, ct1.contours+ct2.contours)

def color_clustering(image, max_iter, kmean_threshold, merge_threshold, k):
    pixels = image.reshape(-1, 3).numpy().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, kmean_threshold)

    _, labels, colors = cv2.kmeans(data=pixels,
                                    K=k,
                                    bestLabels=None,
                                    attempts=10,
                                    criteria=criteria,
                                    flags=cv2.KMEANS_RANDOM_CENTERS)

    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            if np.sum((colors[i] - colors[j])**2) < merge_threshold:
                labels[labels == j] = i
    
    unique_labels = np.unique(labels)
    colors = np.array([np.mean(pixels[(labels == label).squeeze(), :], axis=0) for label in unique_labels])
    for idx, value in enumerate(unique_labels):
        labels[labels == value] = idx

    clustered_pixels = colors[labels.flatten()]
    clustered_image = clustered_pixels.reshape(image.shape)

    labels_matrix = labels.reshape(image.shape[:2])
    return labels_matrix, colors

def labels_to_contour_records(labeled_image, colors, area_threshold, bg_color=None):
    assert len(np.unique(labeled_image)) == np.max(labeled_image) + 1 == len(colors)
    contour_list = []
    forbidden_bg_idx = -1
    
    if bg_color is not None:
        # kick bg out
        color_dis = -1
        for idx, c in enumerate(colors):
            new_dis = np.sum((c - bg_color)**2)
            if color_dis == -1 or new_dis < color_dis:
                forbidden_bg_idx = idx
                color_dis = new_dis
    for i in range(len(colors)):
        # Neglect the bg_color
        if i == forbidden_bg_idx: continue
        color = colors[i]
        binary_image = (labeled_image == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Traverse the single color blocks
        for ct in contours:
            ct_rec = contour_record(color, ct)
            if ct_rec.area > area_threshold:
                contour_list.append(ct_rec)
    return contour_list

def update_lower_layers(G, active_set):
    subgraph = G.subgraph(active_set)
    topological_order = list(nx.topological_sort(subgraph))
    for node in reversed(topological_order):
        successors = list(G.successors(node))
        node.lower_layers = set([item for successor in successors for item in successor.lower_layers] + successors)
    

def visualize_convex_hull_of_contour_list(contour_list, canvas_size, stroke_width, bg_color=None):
    if bg_color is not None:
        image = np.full((canvas_size[0], canvas_size[1], 3), bg_color, dtype=np.uint8)
    else:
        image = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
        
    for record in contour_list:
        color_int = tuple(map(int, record.color*255))
        color_int = color_int[::-1] # reverse it to BGR
        cv2.drawContours(image, [record.convex_hull], -1, color_int, thickness=stroke_width)
    return image

class CycleInDAGError(Exception):
    pass

def initialize(image, cfg, area_coefficient = 1, convex_info = True):
    h, w = image.shape[:-1]
    # Initialize primitive class
    primitive.init_class_vars(w, h, 
                              type_lr=cfg.lr.type,
                              color_lr=cfg.lr.color,
                              triangle_lr=cfg.lr.triangle,
                              rectangle_lr=cfg.lr.rectangle,
                              ellipse_lr=cfg.lr.ellipse,
                              order_lr=cfg.lr.order,
                              print_values=cfg.print_values,
                              print_grads=cfg.print_gradients)
    
    contour_record.boundary_error = cfg.boundary_error
    
    if cfg.bg.color == None:
        bg_color = None
    else:
        bg_color = np.array(cfg.bg.color)

    labeled_img, colors = color_clustering(image, max_iter=cfg.kmeans.max_iter,
                                            kmean_threshold=cfg.kmeans.stop_threshold,
                                            merge_threshold=cfg.kmeans.merge_threshold,
                                            k=cfg.kmeans.start_center_num)
    contours_list = labels_to_contour_records(labeled_img, colors, cfg.area_threshold, bg_color)
    
    if cfg.save.visualize_optimization:
        original_blocks = visualize_convex_hull_of_contour_list(contours_list, (h, w), 2, bg_color)
        cv2.imwrite(cfg.experiment_dir+'/original_blocks.png', original_blocks)
    
    if convex_info:
        # For every edge A->B, it represents that A is over B
        G = nx.DiGraph()
        G.add_nodes_from(contours_list)
        # Compute single occupancy relation
        for i in range(len(contours_list)):
            for j in range(len(contours_list)):
                if j == i: continue
                if contours_list[i].get_erased_by(contours_list[j]) != 0 or contours_list[i].include(contours_list[j]):
                    G.add_edge(contours_list[j], contours_list[i])
                    contours_list[j].lower_layers.add(contours_list[i])
                    
        # Refresh and create a dic for colors
        if any(nx.simple_cycles(G)) == True:
            raise CycleInDAGError("There is a cycle in DAG!")
        color_dic = {}
        active_set = set()
        for i in range(len(contours_list)):
            contours_list[i].update_extra_area()
            active_set.add(contours_list[i])
            key = tuple(contours_list[i].color)
            if key not in color_dic:
                color_dic[key] = [contours_list[i]]
            else:
                color_dic[key].append(contours_list[i])
            
        # Try to merge blocks with the same color
        # color_dic also works like a disjoint set to mark which one is active
        for color, blocks in color_dic.items():
            count = 0
            for i in range(len(blocks)):
                for j in range(i+1, len(blocks)):
                    count += 1

                    if isinstance(blocks[i], int) or isinstance(blocks[j], int):
                        continue
                    
                    if blocks[j] in blocks[i].lower_layers\
                        or blocks[i] in blocks[j].lower_layers:
                            continue
                    
                    tmp = generate_new_contour(blocks[i], blocks[j])
                    forbidden_flag = False
                    potential_edges = []
                    # Get erased by others
                    for other in active_set:
                        if other == blocks[i] or other == blocks[j]: continue
                        a = tmp.get_erased_by(other)
                        if a > 0:
                            if other in blocks[i].lower_layers\
                                or other in blocks[j].lower_layers:
                                    forbidden_flag = True
                                    break
                            # If they have the same color, do not add any constraint.
                            if tuple(other.color) != tuple(color):
                                potential_edges.append((other, blocks[i]))
                                potential_edges.append((other, blocks[j]))
                            
                    if forbidden_flag == True: continue
                    # If some fragements don't merge, increase this threshold or increase the stroke's width
                    if tmp.extra_area > 0: continue
                    
                    # Valid merge
                    # Update equivalent nodes
                    # Here we have already known i_root != j_root
                    blocks[j].merge_from(blocks[i])
                    # Copy paste the edge relations from i_root to j_root
                    for predecessor in G.predecessors(blocks[i]):
                        G.add_edge(predecessor, blocks[j])
                    for successor in G.successors(blocks[i]):
                        G.add_edge(blocks[j], successor)
                    G.add_edges_from(potential_edges)
                    active_set.remove(blocks[i])
                    blocks[i] = j
                    # Only its ancestors and descendants need to be updated
                    ancestors = nx.ancestors(G, blocks[j])
                    descendants = nx.descendants(G, blocks[j])
                    update_lower_layers(G, (ancestors | descendants).add(blocks[j]))
                        
        subgraph = G.subgraph(active_set)
        if cfg.save.visualize_optimization:
            plt.clf()
            pos = nx.spring_layout(subgraph, k=5)
            nx.draw(subgraph, pos, with_labels=True, node_size=100, node_color="lightblue", font_size=10, font_color="black", arrows=True)
            plt.title("Directed Graph")
            plt.savefig(cfg.experiment_dir+"/graph.png")
        
        topological_order = list(nx.topological_sort(subgraph))
        # mapping_to_nums = {i: len(list(comp)) for i, comp in enumerate(strongly_connected_components)}
        # print(mapping_to_nums)
        # print(topological_order)
        
        
    prims = []
    if convex_info:
        depth = 0.0
        for shape in reversed(topological_order):
            prims.append(shape.output_primitive(depth=depth))
            depth -= area_coefficient
    else:
        if cfg.bg.color is not None:
            prims.append(background(color_init=cfg.bg.color, change_color=cfg.bg.optimization))
        sorted_list = sorted(contours_list, key=lambda x: x.area, reverse=True)
        for shape in sorted_list:
            prims.append(shape.output_primitive())
            
    if hasattr(cfg.kmeans, 'max_fragments') and cfg.cut_fragments == True:
        prims = prims[:cfg.kmeans.max_fragments]
    
    return prims