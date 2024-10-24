#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import label2rgb
import numpy as np
import os.path as op
import sh
import csv
import warnings
import pandas as pd
from pims import Bioformats
from PIL import Image
import skimage.filters as sf
import skimage.io as si
import skimage.measure as sme
import skimage.morphology as smo
import scipy.ndimage as sni
import pickle
from tifffile import imwrite

def compute_patches(folder_tif):
    
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif*'))):
        
        print('Loading ... ' + tif_filename)
        filename = (tif_filename.split('.')[0]).split('tiff')[1]
        
        name, raw_tyxc = load_TIF_series(tif_filename)
        
        print('Making masks...')
        (actin_mask_t, zyx_mask_t, 
         vasp_actin_mask_t,vasp_zyx_mask_t, vasp_bg_t) = make_masks(raw_tyxc)
        
        print('Labeling + identifying traces...')
        vasp_zyx_labeled_t = [sme.label(final, connectivity=2)
                     for t, final in enumerate(np.array(vasp_bg_t))]

        vasp_zyx_trace_d, vasp_zyx_regions = identify_traces(vasp_zyx_labeled_t)
        
        print('Computing and saving...')
        compute_properties(vasp_zyx_trace_d, vasp_zyx_regions, raw_tyxc, filename)
        
def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(raw_tyxc, t_start=5, window_size=3):
    
    nt, ny, nx, nc = raw_tyxc.shape
    
    ### generate a rolling average movie to minimize fluctuations
    avg_tyxc = np.zeros((nt, ny, nx, nc))
    for t in range(nt):
        if t < t_start:
            continue

        # Average over a window centered at t.
        t_b = t - window_size // 2
        t_e = t_b + window_size
        t_b = max(t_start, t_b)
        t_e = min(nt, t_e)
        img_yxc = raw_tyxc[t_b:t_e, :, :, :].mean(axis=0)
        
        avg_tyxc[t] = img_yxc
    
    #si.imsave('averaged-data.tiff', avg_tyxc, check_contrast=False)
    
    ###  generate actin mask from averaged movie 
    raw_actin_t = avg_tyxc[:, :, :, 0]
    
    actin_thresh = raw_actin_t > sf.threshold_otsu(raw_actin_t)
    actin_mask_t = smo.remove_small_objects(actin_thresh, 4, 2)
    
    si.imsave('actin.tiff', actin_mask_t, check_contrast=False)
        
    ### generate zyxin mask from background subtracted tiff
    raw_zyx_t = avg_tyxc[:, :, : , 2]
    
    zyx_mask_t = []
    for t, zyx in enumerate(raw_zyx_t):
        if t <= t_start:
            thresh = sf.threshold_yen(raw_zyx_t[t_start])
        else:
            thresh = sf.threshold_yen(zyx)
        mask = zyx > thresh
        zyx_mask_t.append(mask)
    
    actin_zyx_t = zyx_mask_t * actin_mask_t
    clean_zyx_mask_t = []
    for t, zyx in enumerate(actin_zyx_t):
        clean = smo.remove_small_objects(zyx, 4, 2)
        clean_zyx_mask_t.append(clean)
    
    ### keep only persisent zyxin patches
    persistent_t = np.zeros((nt, ny, nx), dtype=bool)
    adjacent_t = np.zeros((nt, ny, nx), dtype=bool)
    for t, zyx in enumerate(clean_zyx_mask_t):
        adjacent = clean_zyx_mask_t[t - 1] if t > 0 else None
        if t + 1 < len(clean_zyx_mask_t):
            adjacent = (adjacent | clean_zyx_mask_t[t + 1]) \
                if adjacent is not None else clean_zyx_mask_t[t + 1]
        adjacent_t[t] = adjacent

        labeled = sme.label(zyx, connectivity=2)
        labels = [reg.label
                  for reg in sme.regionprops(labeled,
                                             intensity_image=adjacent)
                  if reg.max_intensity != 0]
        persistent_t[t] = np.isin(labeled, labels)
    
    #si.imsave('persistent_zyx.tiff', persistent_t, check_contrast=False)
    #si.imsave('zyx-mask.tiff', np.array(zyx_mask_t), check_contrast=False)
    si.imsave('zyx-clean.tiff', np.array(clean_zyx_mask_t), check_contrast=False)
    
    ### generate vasp masks from background subtracted tiffs, and label vasp
    
    raw_vasp_t = avg_tyxc[:, :, : , 1]
    
    vasp_mask_t = []
    for t, vasp in enumerate(raw_vasp_t):
        if t <=t_start:
            thresh = sf.threshold_yen(raw_vasp_t[t_start])
        else:
            thresh = sf.threshold_yen(vasp)
        mask = vasp > thresh
        vasp_mask_t.append(mask)

    vasp_actin_t = vasp_mask_t * actin_mask_t * ~persistent_t
    vasp_zyxactin_t = vasp_mask_t * persistent_t
    vasp_bg_t = vasp_mask_t * ~persistent_t * ~actin_mask_t
    
    clean_vasp_zyxactin_mask_t = []
    for t, vasp in enumerate(vasp_zyxactin_t):
        clean = smo.remove_small_objects(vasp, 4, 2)
        clean_vasp_zyxactin_mask_t.append(clean)
    
    clean_vasp_actin_mask_t = []
    for t, vasp in enumerate(vasp_actin_t):
        clean = smo.remove_small_objects(vasp, 4, 2)
        clean_vasp_actin_mask_t.append(clean)
        
    clean_vasp_actin_t = np.array(clean_vasp_actin_mask_t)
    clean_vasp_zyxactin_t = np.array(clean_vasp_zyxactin_mask_t)
    
    si.imsave('vasp.tiff', np.array(vasp_mask_t), check_contrast=False)
    si.imsave('vasp-actin.tiff', clean_vasp_actin_t, check_contrast=False)
    si.imsave('vasp-zyxactin.tiff', clean_vasp_zyxactin_t, check_contrast=False)

    return (actin_mask_t, persistent_t, clean_vasp_actin_t, 
            clean_vasp_zyxactin_t, vasp_bg_t)

def identify_traces(labeled_t, max_jump=10, remove_static=False):

    """
    Identify dot traces from overlapping labels
    """
    # Every label l in frame t defines a node (t, l). "graph" is a dict such
    # that graph[(t, l)] is a list of nodes that are "connected" to (t, l).
    graph = {}
    regions = {}
    for t, labeled in enumerate(labeled_t):
        for reg in sme.regionprops(labeled):
            l = reg.label
            node = (t, l)

            regions[node] = reg
            graph.setdefault(node, []).append(node)

            y_i, x_i = reg.coords.T  # split the two columns
            for t_prev in range(max(0, t - max_jump), t):
                for l_prev in set(labeled_t[t_prev][y_i, x_i]) - {0}:
                    node_prev = (t_prev, l_prev)
                    graph.setdefault(node, []).append(node_prev)
                    graph.setdefault(node_prev, []).append(node)

    # Perform a depth-first traversal of the graph, recording each connected
    # component as a trace (i.e., a dot). Index traces by _d.
    trace_d = []
    remaining = set(graph.keys())
    while remaining:
        trace = []
        to_visit = [remaining.pop()]
        while to_visit:
            node = to_visit.pop()
            trace.append(node)
            for neighbor in graph[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    to_visit.append(neighbor)
        trace_d.append(sorted(trace))

    # Remove static dots, i.e., traces that persist to the last frame.
    if remove_static:
        trace_d = [
            tr for tr in trace_d if tr[-1][0] != len(labeled_t) - 1]

    return trace_d, regions

def compute_properties(trace_d, regions, raw_ctyx, fn, channel=2):
    
    # Compute lifetime of each dot.
    lifetime_d = [tr[-1][0] - tr[0][0] + 1
                  for tr in trace_d]
            
    # Extract coordinates for each node
    coords_d = []
    for trace in trace_d:
        coords_n = [regions[node].coords for node in trace]
        coords_d.append(coords_n)

    # Extract centroid for each node
    centroid_d = []
    for trace in trace_d:
        centroid_n = [regions[node].centroid for node in trace]
        centroid_d.append(centroid_n)

    # Compute the local intensity ratio of each dot
    nt, ny, nx, nc = raw_ctyx.shape
    intensity_ratio_max_d = []
    for trace, lifetime in zip(trace_d, lifetime_d):
        intensity_ratio_n = []
        for node in trace:
            reg = regions[node]
            y_i, x_i = reg.coords.T  # split the two columns
            radius = max(1, int(2 * np.sqrt(reg.area)))

            inner = np.zeros((ny, nx), dtype=bool)
            inner[y_i, x_i] = 1
            outer = smo.binary_dilation(inner, smo.disk(radius)) & ~inner
            
            image = raw_ctyx[node[0], :, :, channel]
            inner_mean = image[inner].mean()
            outer_mean = image[outer].mean()
            intensity_ratio_per_node = inner_mean / outer_mean
            intensity_ratio_n.append(intensity_ratio_per_node)

        intensity_ratio_max_d.append(np.max(intensity_ratio_n))

        '''#print('%-3d  %.4f' % (lifetime, intensity_ratio_max_d[-1]))'''
    
    # Find lifetime, area, major axis length, aspect ratio, and eccentricity of each dot
    area_max_d = []
    major_axis_length_max_d = []
    aspect_ratio_max_d = []
    eccentricity_max_d = []
    for d, trace in enumerate(trace_d):
        area_n = [regions[node].area for node in trace]
                
        major_axis_length_n = [regions[node].major_axis_length for node in trace]
        eccentricity_n = [regions[node].eccentricity for node in trace]
        aspect_ratio_n = [(regions[node].major_axis_length / 
                          (regions[node].area / regions[node].major_axis_length)) 
                              for node in trace]
            
        area_max_d.append(np.max(area_n))
        major_axis_length_max_d.append(np.max(major_axis_length_n))
        aspect_ratio_max_d.append(np.max(aspect_ratio_n))
        eccentricity_max_d.append(np.max(eccentricity_n))

    output = dict(
        trace_d=trace_d,
        lifetime_d=lifetime_d,
        intensity_ratio_max_d=intensity_ratio_max_d,
        area_max_d=area_max_d,
        major_axis_length_max_d=major_axis_length_max_d,
        aspect_ratio_max_d=aspect_ratio_max_d,
        eccentricity_max_d=eccentricity_max_d,
        coords_d=coords_d,
        centroid_d=centroid_d,
    )
    
    output_txt = ('output/' + fn + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for d, lifetime in enumerate(lifetime_d):
            f.write('%-3d  %.4f  %.4f  %.4f %.4f  %4f\n' % (
                lifetime, intensity_ratio_max_d[d], area_max_d[d],
                major_axis_length_max_d[d], aspect_ratio_max_d[d], eccentricity_max_d[d]))
    
    return output

def filter_props(folder_txt, lifetime_cutoff=5, ir_min=1, ir_max = 1000):
    
    filename = (folder_txt.split('/')[1])
    output_txt = ('output/filtered/' + filename + '.txt')    
    print(output_txt)

    lifetime_d = []
    intensity_ratio_max_d = []
    area_max_d = []
    aspect_ratio_max_d = []
    
    for file in sorted(glob(op.join(folder_txt, '*.txt'))):
        
        print('Loading: ' + file)
        
        with open(file, 'r') as f:
            lines=f.readlines()
            for x in lines:
                unclean_split = x.split(' ')
                clean_split = []
                for item in unclean_split:
                    if len(item) >= 1:
                        clean_split.append(float(item))
                
                lifetime_d.append(clean_split[0])
                intensity_ratio_max_d.append(clean_split[1])
                area_max_d.append(clean_split[2])
                aspect_ratio_max_d.append(clean_split[4])
            f.close()

    lifetime_D = []
    intensity_ratio_max_D = []
    area_max_D = []
    aspect_ratio_max_D = []
    for (lifetime, intensity_ratio_max, area_max,
         aspect_ratio_max) in zip(
             lifetime_d, intensity_ratio_max_d, area_max_d, aspect_ratio_max_d):
        if (lifetime > lifetime_cutoff) and (ir_max > intensity_ratio_max >
                                             ir_min):
            lifetime_D.append(lifetime)
            intensity_ratio_max_D.append(intensity_ratio_max)
            area_max_D.append(area_max)
            aspect_ratio_max_D.append(aspect_ratio_max)
            
    # Convert number of frames to actual lifetime.
    # Imaging interval: 1.17s
    lifetime_D = [lifetime * 1.17 for lifetime in lifetime_D]

    # Convert number of pixels to micron2. Pixel size: 0.267 um.
    area_max_D = [area_max * 0.0256 for area_max in area_max_D]
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for D, lifetime in enumerate(lifetime_D):
            f.write('%-3d  %.4f  %.4f  %.4f\n' % (
                lifetime, intensity_ratio_max_D[D], area_max_D[D],
                aspect_ratio_max_D[D]))

def count_events(folder_txt, lifetime_cutoff=0, ir_min=0, ir_max = 1000):
    
    filename = (folder_txt.split('/')[1])
    output_txt = ('output/count_n_filtered/' + filename + '.txt')    
    print(output_txt)
    
    events = []
    
    for file in sorted(glob(op.join(folder_txt, '*.txt'))):
        lifetime_d = []
        intensity_ratio_max_d = []
        area_max_d = []
        aspect_ratio_max_d = []
        
        print('Loading: ' + file)
        
        with open(file, 'r') as f:
            lines=f.readlines()
            for x in lines:
                print(x)
                unclean_split = x.split(' ')
                clean_split = []
                for item in unclean_split:
                    if len(item) >= 1:
                        clean_split.append(float(item))
                
                lifetime_d.append(clean_split[0])
                intensity_ratio_max_d.append(clean_split[1])
                area_max_d.append(clean_split[2])
                aspect_ratio_max_d.append(clean_split[5])
            f.close()

        lifetime_D = []
        intensity_ratio_max_D = []
        area_max_D = []
        aspect_ratio_max_D = []
        for (lifetime, intensity_ratio_max, area_max,
             aspect_ratio_max) in zip(
                 lifetime_d, intensity_ratio_max_d, area_max_d, aspect_ratio_max_d):
            if ir_max > intensity_ratio_max > ir_min:
                lifetime_D.append(lifetime)
                intensity_ratio_max_D.append(intensity_ratio_max)
                area_max_D.append(area_max)
                aspect_ratio_max_D.append(aspect_ratio_max)
            
        total_events = len(lifetime_D)
        events.append(total_events)
        
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for item in events:
            f.write('%-3d\n' % (item))

def consolidate_props(folder_txt, lifetime_cutoff=0, ir_min=0, ir_max = 1000):
    
    filename = (folder_txt.split('/')[1])
    output_txt = ('output/filtered/' + filename + '.txt')    
    print(output_txt)

    lifetime_d = []
    
    for file in sorted(glob(op.join(folder_txt, '*.txt'))):
        
        print('Loading: ' + file)
        
        with open(file, 'r') as f:
            lines=f.readlines()
            for x in lines:
                lifetime_d.append(float(x))
            f.close()
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for D, lifetime in enumerate(lifetime_d):
            f.write('%-3d\n' % (lifetime))
            
            
if __name__ == '__main__':
    for folder_tif in sorted(glob('tiff/*')):
        compute_patches(folder_tif)
    
    #for folder_txt in sorted(glob('output/*')):
        #filter_props(folder_txt)
            #count_events(folder_txt)
