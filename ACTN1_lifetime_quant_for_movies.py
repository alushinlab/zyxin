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
import itertools
from tifffile import imwrite

def compute_lifetime(folder_tif):
    
    max_ratio = []
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif*'))):
        
        prefix = (tif_filename.split('/')[1]).split('_')[0] + '_'
        filename = prefix + (tif_filename.split('/')[2]).split('.tif')[0]
        print('Loading ... ' + filename)
        
        name, img_tyxc = load_TIF_series(tif_filename)
    
        print('Making masks...')
        (zyx_actn1, zyx_noactn1, actn1_nozyx, actn1_mask, zyx_mask) = make_masks(img_tyxc)
        
        print('Labeling + identifying traces...')
        labeled = [sme.label(final, connectivity=2)
                     for t, final in enumerate(np.array(actn1_nozyx))]
        
        #trace_d, regions = identify_traces(labeled)
        
        print('Computing and saving...')
        #compute_properties(trace_d, regions, img_tyxc, filename)
        
        ratio = compute_properties_whole(actn1_nozyx, actn1_mask, img_tyxc, filename, t_start=5)
        
        max_ratio.append(ratio)
    
    print(max_ratio)

def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(img_tyxc, t_start=5, window_size=3):
    
    '''
    make masks of each channel
    for actn1 thresholding, use yen for ABM, and li for WT
    for zyxin thresholding, use yen for noATP, and otsu for ATP bc of patches
    '''
    
    nt, ny, nx, nc = img_tyxc.shape

    # generate a rolling average movie to minimize fluctuations
    avg_tyxc = np.zeros((nt, ny, nx, nc))
    for t in range(nt):
        if t < t_start:
            continue
        # Average over a window centered at t.
        t_b = t - window_size // 2
        t_e = t_b + window_size
        t_b = max(t_start, t_b)
        t_e = min(nt, t_e)
        avg_tyxc[t] = img_tyxc[t_b:t_e, :, :, :].mean(axis=0)

    #si.imsave('averaged-data.tiff', avg_tyxc, check_contrast=False) 
    
    # generate actin mask
    avg_actin = avg_tyxc[:, :, :, 0]
    
    actin_thresh = avg_actin > sf.threshold_li(avg_actin)
    actin_mask = np.array(smo.remove_small_objects(actin_thresh, 5, 2))
    si.imsave('actin.tiff', actin_mask, check_contrast=False)

    # generate zyxin mask using sobel------------------------------------------
    avg_zyx = avg_tyxc[:, :, :, 2]
    zyx_mask = []
    for t, zyx in enumerate(avg_zyx):
        sobel = sf.sobel(zyx)
        # use otsu for zyxin ATP patches, but yen for noATP punctas
        thresh = sf.threshold_otsu(sobel)
        bright = smo.binary_opening(sobel > thresh)
        filled = sni.binary_fill_holes(bright)
        mask = smo.binary_erosion(filled)
        zyx_mask.append(mask)
    zyx_actin = zyx_mask * actin_mask
    
    # remove any zyxin not colocalized with actin
    zyx_clean = []
    for t, zyx in enumerate(zyx_actin):
        clean = smo.remove_small_objects(zyx, 3, 2)
        zyx_clean.append(clean)
        
    # keep only persistent zyxin events        
    persistent = np.zeros((nt, ny, nx), dtype=bool)
    adjacent_t = np.zeros((nt, ny, nx), dtype=bool)
    for t, zyx in enumerate(zyx_clean):
        adjacent = zyx_clean[t - 1] if t > 0 else None
        if t + 1 < len(zyx_clean):
            adjacent = (adjacent | zyx_clean[t + 1]) \
                if adjacent is not None else zyx_clean[t + 1]
        adjacent_t[t] = adjacent    
        
        labeled = sme.label(zyx, connectivity=2)
        labels = [reg.label
                  for reg in sme.regionprops(labeled,
                                             intensity_image=adjacent)
                  if reg.max_intensity != 0]
        persistent[t] = np.isin(labeled, labels)

    si.imsave('zyx-mask.tiff', np.array(persistent), check_contrast=False)

    # generate actn1 masks ----------------------------------------------------
    
    avg_actn1 = avg_tyxc[:, :, :, 1]
    actn1_mask = []
    for t, actn1 in enumerate(avg_actn1):
        sobel = sf.sobel(actn1)
        # if its WT ACTN1, use li for softer thresholding. For ABM, use yen
        thresh = sf.threshold_li(sobel)
        bright = smo.binary_opening(sobel > thresh)
        filled = sni.binary_fill_holes(bright)
        mask = smo.binary_erosion(filled)
        actn1_mask.append(mask)
    actn1_actin = actn1_mask * actin_mask
    
    actn1_clean = []
    for t, actn1 in enumerate(actn1_actin):
        clean = smo.remove_small_objects(actn1, 5, 2)
        actn1_clean.append(clean)
        
    si.imsave('actn1-mask.tiff', np.array(actn1_clean), check_contrast=False)
    
    # generate zyxin colocalized with actn1, or not
    zyx_actn1 = np.array(persistent) * np.array(actn1_clean)    
    zyx_noactn1 = np.array(persistent) * ~np.array(actn1_clean)
    actn1_nozyx= ~np.array(persistent) * np.array(actn1_clean) 
    
    si.imsave('zyx_actn1.tiff', np.array(zyx_actn1), check_contrast=False)
    #si.imsave('zyx_noactn1.tiff', np.array(zyx_noactn1), check_contrast=False)
    
    return(zyx_actn1, zyx_noactn1, actn1_nozyx, np.array(actn1_mask), persistent)

def identify_traces(labeled_t, max_jump=5, remove_static=False):

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
    
def compute_properties(trace_d, regions, raw_ctyx, fn):
    
    # Compute lifetime of each dot.
    lifetime_d = [tr[-1][0] - tr[0][0] + 1
                  for tr in trace_d]
    
    print(len(trace_d))
            
    # Extract coordinates for each node
    coords_d = []
    for trace in trace_d:
        coords_n = [regions[node].coords for node in trace]
        coords_d.append(coords_n)

    # Compute the local intensity ratio of each dot
    nt, ny, nx, nc = raw_ctyx.shape
    intensity1_ratio_max_d = []
    intensity2_ratio_max_d = []
    for trace, lifetime in zip(trace_d, lifetime_d):
        intensity1_ratio_n = []
        intensity2_ratio_n = []
        for node in trace:
            reg = regions[node]
            y_i, x_i = reg.coords.T  # split the two columns
            radius = max(1, int(2 * np.sqrt(reg.area)))

            inner = np.zeros((ny, nx), dtype=bool)
            inner[y_i, x_i] = 1
            outer = smo.binary_dilation(inner, smo.disk(radius)) & ~inner
            
            # calculate channel 1 intensity (actn1)
            image = raw_ctyx[node[0], :, :, 1]
            inner_mean = image[inner].mean()
            outer_mean = image[outer].mean()
            intensity_ratio_per_node = inner_mean / outer_mean
            intensity1_ratio_n.append(intensity_ratio_per_node)
            
            # calculate channel 2 intensity (Zyxin)
            image = raw_ctyx[node[0], :, :, 2]
            inner_mean = image[inner].mean()
            outer_mean = image[outer].mean()
            intensity_ratio_per_node = inner_mean / outer_mean
            intensity2_ratio_n.append(intensity_ratio_per_node)

        intensity1_ratio_max_d.append(np.max(intensity1_ratio_n))
        intensity2_ratio_max_d.append(np.max(intensity2_ratio_n))
    
    # Find lifetime, area, major axis length, aspect ratio, and eccentricity of each dot
    area_max_d = []
    major_axis_length_max_d = []
    aspect_ratio_max_d = []
    #eccentricity_max_d = []
    for d, trace in enumerate(trace_d):
        area_n = [regions[node].area for node in trace]
                
        major_axis_length_n = [regions[node].major_axis_length for node in trace]
        #eccentricity_n = [regions[node].eccentricity for node in trace]
        aspect_ratio_n = [(regions[node].major_axis_length / 
                          (regions[node].area / regions[node].major_axis_length)) 
                              for node in trace]
            
        area_max_d.append(np.max(area_n))
        major_axis_length_max_d.append(np.max(major_axis_length_n))
        aspect_ratio_max_d.append(np.max(aspect_ratio_n))
        #eccentricity_max_d.append(np.max(eccentricity_n))

    output = dict(
        trace_d=trace_d,
        lifetime_d=lifetime_d,
        intensity1_ratio_max_d=intensity1_ratio_max_d,
        intensity2_ratio_max_d=intensity2_ratio_max_d,
        area_max_d=area_max_d,
        major_axis_length_max_d=major_axis_length_max_d,
        aspect_ratio_max_d=aspect_ratio_max_d,
        #eccentricity_max_d=eccentricity_max_d,
        coords_d=coords_d,
    )
    
    output_txt = ('output/' + fn + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for d, lifetime in enumerate(lifetime_d):
            f.write('%-3d  %.4f  %.4f  %.4f %.4f  %4f\n' % (
                lifetime, intensity1_ratio_max_d[d], intensity2_ratio_max_d[d], 
                area_max_d[d], major_axis_length_max_d[d], aspect_ratio_max_d[d]))
    
    return output

def compute_properties_whole(mask_t, zyx_mask, raw_ctyx, fn, t_start=5):
    
    actn1_intensity_ratio = []

    for t, mask in enumerate(mask_t):
        if t < t_start:
            continue
        
        # dilate actn1 mask for bg ratio
        mask_area = np.sum(mask)
        dilated = smo.binary_dilation(mask) * ~zyx_mask * ~mask
        dilated_area = np.sum(dilated)

        image = raw_ctyx[t, :, :, 1]
        inner = np.sum(image * mask) / mask_area
        outer = np.sum(image * dilated) / dilated_area
        intensity_ratio = inner / outer
        actn1_intensity_ratio.append(intensity_ratio)

    max_ratio = (np.max(actn1_intensity_ratio))
    
    return(max_ratio)

def filter_props(folder_txt, lifetime_cutoff=1, ir_min=1.5, ir_max = 1000):
    
    filename = (folder_txt.split('/')[1])
    output_txt = ('output/filtered/' + filename + '.txt')    
    print(output_txt)

    lifetime_D = []
    intensity1_ratio_max_D = []
    intensity2_ratio_max_D = []
    area_max_D = []
    aspect_ratio_max_D = []
    
    for file in sorted(glob(op.join(folder_txt, '*.txt'))):
        
        print('Loading: ' + file)
        
        lifetime_d = []
        intensity1_ratio_max_d = []
        intensity2_ratio_max_d = []
        area_max_d = []
        aspect_ratio_max_d = []
        
        with open(file, 'r') as f:
            lines=f.readlines()
            for x in lines:
                unclean_split = x.split(' ')
                clean_split = []
                for item in unclean_split:
                    if len(item) >= 1:
                        clean_split.append(float(item))
                
                lifetime_d.append(clean_split[0])
                intensity1_ratio_max_d.append(clean_split[1])
                intensity2_ratio_max_d.append(clean_split[2])
                area_max_d.append(clean_split[3])
                aspect_ratio_max_d.append(clean_split[5])
            f.close()
        
        events = 0
        
        for (lifetime, intensity1_ratio_max, intensity2_ratio_max, area_max,
             aspect_ratio_max) in zip(
                 lifetime_d, intensity1_ratio_max_d, intensity2_ratio_max_d, 
                 area_max_d, aspect_ratio_max_d):
            
            if (lifetime > lifetime_cutoff) and (
            ir_max > intensity1_ratio_max > 1.3) and (
            ir_max > intensity2_ratio_max > ir_min):
                lifetime_D.append(lifetime)
                intensity1_ratio_max_D.append(intensity1_ratio_max)
                intensity2_ratio_max_D.append(intensity2_ratio_max)
                area_max_D.append(area_max)
                aspect_ratio_max_D.append(aspect_ratio_max)
                events += 1
            
        print('number of events:' + str(events))
            
    # Convert number of frames to actual lifetime.
    # Imaging interval: 1.17s
    lifetime_D = [lifetime * 3 for lifetime in lifetime_D]

    # Convert number of pixels to micron2. Pixel size: 0.267 um.
    area_max_D = [area_max * 0.0256 for area_max in area_max_D]
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for D, lifetime in enumerate(lifetime_D):
            f.write('%-3d  %.4f  %.4f  %.4f  %.4f\n' % (
                lifetime, intensity1_ratio_max_D[D], intensity2_ratio_max_D[D],
                area_max_D[D], aspect_ratio_max_D[D]))
            
if __name__ == '__main__':
    for folder_tif in sorted(glob('tiff/m*')):
        print(' ---- ' + folder_tif + ' --- ')
        compute_lifetime(folder_tif)
    
    #for folder_txt in sorted(glob('output/*')):
        #filter_props(folder_txt)
