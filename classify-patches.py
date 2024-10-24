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
import itertools
from tifffile import imwrite
from collections import Counter

def compute_patches(folder_tif):
    
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif*'))):
        
        filename = (tif_filename.split('/')[2]).split('.tif')[0]
       
        print('Loading ... ' + filename)
        
        name, raw_tyxc = load_TIF_series(tif_filename)
        
        print('Making masks...')
        (actin_mask_t, zyx_mask_t) = make_masks(raw_tyxc, filename)
        
        print('Labeling + identifying traces...')
        labeled_t = [sme.label(final, connectivity=2)
                     for t, final in enumerate(np.array(zyx_mask_t))]

        zyx_trace_d, zyx_regions = identify_traces(labeled_t)
        
        print('Computing and saving...')
        patch_sequence_d = compute_properties(zyx_trace_d, zyx_regions,raw_tyxc, actin_mask_t, filename, channel=1, threshold=0.8)
        
        print('Sequence Order...')
        count_order(patch_sequence_d, filename)
        
def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(raw_tyxc, fn, t_start=3, window_size=3):
    print('Finding patches')
    print(raw_tyxc.shape)

    nt, nc, nx, ny = raw_tyxc.shape

    # Average over window_size and sobel
    sobel_tyx = np.zeros((nt, nx, ny))
    actin_tyx = np.zeros((nt, nx, ny))
    for t in range(nt):
        if t < t_start:
            continue

        # Average over a window centered at t.
        t_b = t - window_size // 2
        t_e = t_b + window_size
        t_b = max(t_start, t_b)
        t_e = min(nt, t_e)
        img_yx = raw_tyxc[t_b:t_e, 1, :, :].mean(axis=0)
        actin_yx = raw_tyxc[t_b:t_e, 0, :, :].mean(axis=0)
        actin_tyx[t] = actin_yx
        # Smooth the LIM channel to denoise
        img_yx = sf.gaussian(img_yx, sigma=1)
        # Perform Sobel transformation
        sobel_tyx[t] = img_yx
    
    # Generate actin mask from averaged movie 
    
    actin_thresh = actin_tyx > sf.threshold_li(actin_tyx)
    actin_mask_t = smo.remove_small_objects(actin_thresh, 4, 2)
    
    actin_mask_txt = ('output/masks/' + fn + '_actin.tif')
    sh.mkdir('-p', op.dirname(op.abspath(actin_mask_txt)))
    si.imsave(actin_mask_txt, np.array(actin_mask_t), check_contrast=False)
    
    # Find bright zyxin patches on each frame
    patch_tyx = np.zeros((nt, ny, nx), dtype=bool)
    for t, sobel_yx in enumerate(sobel_tyx):
        if t < t_start:
            continue

        thresh = sf.threshold_yen(sobel_yx)
        bright_yx = smo.binary_opening(sobel_yx > thresh)
        patch_yx = sni.binary_fill_holes(bright_yx)
        patch_tyx[t] = patch_yx
        
    # Bandpass filter zyxin patches
    filtered_t = np.zeros((nt, ny, nx), dtype=bool)
    for t, zyx in enumerate(patch_tyx):
        filtered_t[t] = bandpass_filter(zyx)
    
    # Keep only persistent zyxin patches
    persistent_t = np.zeros((nt, ny, nx), dtype=bool)
    adjacent_t = np.zeros((nt, ny, nx), dtype=bool)
    
    for t, zyx in enumerate(filtered_t):
        adjacent = filtered_t[t - 1] if t > 0 else None
        if t + 1 < len(filtered_t):
            adjacent = (adjacent | filtered_t[t + 1]) \
                if adjacent is not None else filtered_t[t + 1]
        adjacent_t[t] = adjacent

        labeled = sme.label(zyx, connectivity=2)
        labels = [reg.label
                  for reg in sme.regionprops(labeled,
                                             intensity_image=adjacent)
                  if reg.max_intensity != 0]
        persistent_t[t] = np.isin(labeled, labels)
    
    patch_mask_txt = ('output/masks/' + fn + '_patch.tif')
    sh.mkdir('-p', op.dirname(op.abspath(patch_mask_txt)))
    si.imsave(patch_mask_txt, np.array(persistent_t), check_contrast=False)
    
    return actin_mask_t, persistent_t

def bandpass_filter(label_image, min_size=10, max_size=150):
    small_removed = smo.remove_small_objects(label_image, min_size, connectivity=2)
    mid_removed = smo.remove_small_objects(small_removed, max_size, connectivity=2)
    large_removed = small_removed & ~mid_removed
    return large_removed

def identify_traces(labeled_t, max_jump=3, remove_static=False):

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

def compute_properties(trace_d, regions, raw_ctyx, mask, fn, channel, threshold):
    
    lifetime_d = []
    max_intensity_ratio_zyxin_d = []
    area_d = []
    intensity_ratio_zyx_d = []
    overlap_actin_d = []
    actin_count_d = []
    major_axis_length_d = []
    
    for d, trace in enumerate(trace_d):
    
        lifetime_n = []
        max_intensity_ratio_zyxin_n = []
        area_n = []
        intensity_ratio_zyx_n = []
        overlap_actin_n = []
        actin_count_n = []
        major_axis_length_n = []

        # Compute lifetime of each dot.
        lifetime = trace[-1][0] - trace[0][0] + 1
        #print('lifetime: ' + str(lifetime))
        if lifetime > 1:
            lifetime_n.append(lifetime)

            nt, nc, nx, ny = raw_ctyx.shape
            for node in trace:
                
                area_n.append(regions[node].area)
                major_axis_length_n.append(regions[node].major_axis_length)
                
                reg = regions[node]
                y_i, x_i = reg.coords.T  # split the two columns
                inner = np.zeros((ny, nx), dtype=bool)
                inner[y_i, x_i] = 1
                outer = sni.binary_dilation(inner, iterations=3) & ~inner
                
                # intensity ratio for zyxin
                image = raw_ctyx[node[0], channel, :, :]
                inner_mean = image[inner].mean()
                outer_mean = image[outer].mean()
                intensity_ratio_per_node = inner_mean / outer_mean
                
                intensity_ratio_zyx_n.append(intensity_ratio_per_node)
                max_intensity_ratio_zyxin_n.append(np.max(intensity_ratio_zyx_n))
                
                # fraction overlapping with actin
                zyxactin = inner * mask[node[0], :, :]
                zyxactin_area = np.sum(zyxactin)
                inner_area = np.sum(inner)
                overlap_actin = zyxactin_area / inner_area
                
                overlap_actin_n.append(overlap_actin)
                
                # Count how many objects in patch x actin
                labeled = sme.label(zyxactin)
                labeled_props = sme.regionprops(labeled)
                object_count = len(labeled_props)
                
                actin_count_n.append(object_count)
                
                
                '''# plot image of mask
                f, ax = plt.subplots(1,3) 
                ax[0].imshow(inner)
                ax[0].set_title('patch')
                ax[1].imshow(mask[node[0], :, :])
                ax[1].set_title('actin')
                ax[2].imshow(zyxactin)
                ax[2].set_title('patch x actin')
                plt.show()
                
                
                print('object count:  ' + str(object_count) +
                                ' overlap area: ' + str(overlap_actin))'''
                
        # Only save traces that have overlap with actin
        zeros = actin_count_n.count(0)
        percent_zeros = zeros / len(actin_count_n)
        
        if percent_zeros < 0.25:
            
            lifetime_d.extend(lifetime_n)
            max_intensity_ratio_zyxin_d.extend(max_intensity_ratio_zyxin_n)
            area_d.append(area_n)
            major_axis_length_d.append(major_axis_length_n)
            intensity_ratio_zyx_d.append(intensity_ratio_zyx_n)
            overlap_actin_d.append(overlap_actin_n)
            actin_count_d.append(actin_count_n)
            
    # eliminate minor zeroes maintain continuous segments
    # for easier classification
    actin_count_dz = eliminate_zeroes(actin_count_d)
    
    #print('before: ' , actin_count_dz)
    
    actin_count_df = remove_islands(actin_count_dz)
    while True:
        new_modified_list = remove_islands(actin_count_df)
        if new_modified_list == actin_count_df:
            break
        actin_count_df = new_modified_list
 
    #print('remove islands: ' , actin_count_df)
    
    # separate the bridges from actin-bound-patches/tails
    segmented_actin_count_z = separate_segments(actin_count_df)
    
    # apply segmentation
    
    segmented_length_z= apply_segmentation_dz(segmented_actin_count_z, major_axis_length_d)
    intensity_ratio_zyx_z= apply_segmentation_dz(segmented_actin_count_z, intensity_ratio_zyx_d)
    overlap_actin_z = apply_segmentation_dz(segmented_actin_count_z, overlap_actin_d)
    
    # Classify bridges, tails, and actin-bound patches. 
    max_bridge_length_n = []
    max_bridge_intensity_n = []
    bridge_lifetime_n = []
    max_tail_length_n = []
    max_tail_intensity_n = []
    tail_lifetime_n = []
    max_ab_patch_length_n = []
    max_ab_patch_intensity_n = []
    ab_patch_lifetime_n = []
    
    patch_sequence_d = []

    for d, segmented_actin_count_d in enumerate(segmented_actin_count_z):
        
        patch_sequence = []
        
        for n, actin_count_n in enumerate(segmented_actin_count_d):
            
            if len(actin_count_n) >= 3:
            
                actin_count = Counter(actin_count_n).most_common(1)[0][0]
                
                if actin_count == 2:
                    bridge_lifetime = len(actin_count_n)
                    bridge_lifetime_n.append(bridge_lifetime)
                    max_bridge_length = max(segmented_length_z[d][n])
                    max_bridge_length_n.append(max_bridge_length)
                    max_bridge_intensity = max(intensity_ratio_zyx_z[d][n])
                    max_bridge_intensity_n.append(max_bridge_intensity)
                    print('bridge lifetime: ', bridge_lifetime, 'length: ', max_bridge_length)
                    
                    for i, actin_count in enumerate(actin_count_n):
                        patch_sequence.append('b')
                
                if actin_count == 1:
                    
                    # segment tails and actin-bound patches based on an overlap actin threshold
                    
                    segmented_overlap_a = segment_floats(overlap_actin_z[d][n], threshold)
        
                    if len(actin_count_n) >= 3:
                        
                        # apply segmentation
                        
                        segmented_length_g= apply_segmentation_ng(segmented_overlap_a, segmented_length_z[d][n])
                        intensity_ratio_zyx_g= apply_segmentation_ng(segmented_overlap_a, intensity_ratio_zyx_z[d][n])
                        
                        for a, segment in enumerate(segmented_overlap_a):
                            
                            if len(segment) >= 3:
                                zeroes = segment.count(0)
                                
                                actin_overlap = sum(segment)/(len(segment) - zeroes)
                                
                                
                                if 0 < actin_overlap <= threshold:
                                    print('tail actin overlap: ', actin_overlap)
                                    
                                    tail_lifetime = len(segment)
                                    tail_lifetime_n.append(tail_lifetime)
                                    max_tail_length = max(segmented_length_g[a])
                                    max_tail_length_n.append(max_tail_length)
                                    max_tail_intensity = max(intensity_ratio_zyx_g[a])
                                    max_tail_intensity_n.append(max_tail_intensity)
                                
                                    print('tail lifetime: ', tail_lifetime, 'length: ', max_tail_length)
                                    
                                    for i, actin_count in enumerate(actin_count_n):
                                        patch_sequence.append('t')
                                    
                                    
                                if actin_overlap > threshold:
                                    print('actin-bound patch actin overlap: ', actin_overlap)
                                    
                                    ab_patch_lifetime = len(segment)
                                    ab_patch_lifetime_n.append(ab_patch_lifetime)
                                    max_ab_patch_length = max(segmented_length_g[a])
                                    max_ab_patch_length_n.append(max_ab_patch_length)
                                    max_ab_patch_intensity = max(intensity_ratio_zyx_g[a])
                                    max_ab_patch_intensity_n.append(max_ab_patch_intensity)
                                    
                                    for i, actin_count in enumerate(actin_count_n):
                                        patch_sequence.append('a')
                                
                                    print('actin-bound patch lifetime: ', ab_patch_lifetime, 'length: ', max_ab_patch_length)
        
        if len(patch_sequence) > 0:           
            patch_sequence_d.append(patch_sequence)
                        
    #print('--------- max lifetimes --------')
    #print('actin bound patch: ', max(ab_patch_lifetime_n), 'bridge: ' , max(bridge_lifetime_n), 'tail: ', max(tail_lifetime_n))
    
    save_file(fn, 'f1_bridge', bridge_lifetime_n, max_bridge_length_n, max_bridge_intensity_n)
    save_file(fn, 'f1_tail', tail_lifetime_n, max_tail_length_n, max_tail_intensity_n)
    save_file(fn, 'f1_actin-bound-patch', ab_patch_lifetime_n, max_ab_patch_length_n, max_ab_patch_intensity_n)
    
    return(patch_sequence_d)

def eliminate_zeroes(actin_count_d):
    
    for d, actin_count_n in enumerate(actin_count_d):
            
       old = str(actin_count_n.count(0))

       for n, actin_count in enumerate(actin_count_n):
           
           if n == 0 and actin_count == 0:
               actin_count_n[n] = actin_count_n[n + 1]
            
           if n > 0 and actin_count == 0:
               actin_count_n[n] = actin_count_n[n - 1]
               
           # for simplicity of bridges
           if n > 0 and actin_count > 2:
               actin_count_n[n] = 2
        
       new = str(actin_count_n.count(0))
       #print('before eliminating zeroes: ' + old, 'after: ' + new)
        
    return actin_count_d

def remove_islands(input_list_d):

    modified_list_d = []
    
    for input_list in input_list_d:
        
        modified_list = []
        
        for i, num in enumerate(input_list):
            if i == 0 or i == len(input_list) - 1:
                # For the first and last elements, simply append them to the modified list
                modified_list.append(num)
            elif input_list[i - 1] == num or input_list[i + 1] == num:
                # If the current element has a neighbor with the same value, append it to the modified list
                modified_list.append(num)
            else:
                # If the current element is an island, replace it with its neighboring value
                modified_list.append(input_list[i - 1])
                
        modified_list_d.append(modified_list)

    return modified_list_d

def separate_segments(input_list_d):
    
    segments_d = []
    #before = 0
    #after = 0
    for input_list in input_list_d:
        segments = []
        current_segment = []
    
        for num in input_list:
            #before+=num
            if not current_segment or num == current_segment[-1]:
                current_segment.append(num)
                #after+=num
            else:
                segments.append(current_segment)
                current_segment = [num]
                #after+=num
    
        segments.append(current_segment)
        segments_d.append(segments)

    #print('before', input_list_d)
    #print('after', segments_d)

    return segments_d

def apply_segmentation_dz(segmented_list_z, original_list):
    
    #print('before: ' , original_list)
    
    segmented_original_list_z = []
    
    for n, segmented_list_d in enumerate(segmented_list_z):
        #print(segmented_list_d)
        segmented_original_list_d = []
        
        for segment in segmented_list_d:

            segment_length = len(segment)
            associated_segment = original_list[n][:segment_length]
            removed_segment = original_list[n][segment_length:]
            segmented_original_list_d.append(associated_segment)
            #print((original_list[n][:segment_length]))
            #print('segment: ' , segment_length, 'associated segment: ' , len(associated_segment))
            #print(removed_segment)
            original_list[n] = removed_segment
        
        segmented_original_list_z.append(segmented_original_list_d)

    #print ('after: ' , segmented_original_list_z)
        
    return segmented_original_list_z
                    
def segment_floats(float_list, threshold):
    
    segments = []
    current_segment = []
    above_threshold = float_list[0] >= threshold
    for num in float_list:
        if (num >= threshold) != above_threshold:
            segments.append(current_segment)
            current_segment = []
            above_threshold = not above_threshold
        current_segment.append(num)

    if current_segment:
        segments.append(current_segment)
    
    print(len(segments))

    return segments
                
def apply_segmentation_ng(segmented_list_n, original_list):
    
    #print('before: ' , original_list)
    
    segmented_original_list_g = []
    
    for segment in segmented_list_n:

        segment_length = len(segment)
        associated_segment = original_list[:segment_length]
        removed_segment = original_list[segment_length:]
        segmented_original_list_g.append(associated_segment)
        #print((original_list[n][:segment_length]))
        #print('segment: ' , segment_length, 'associated segment: ' , len(associated_segment))
        #print(removed_segment)
        original_list = removed_segment
        
    #print ('after: ' , segmented_original_list_g)
        
    return segmented_original_list_g

def save_file(filename, patch_type, lifetime_d, length_d, intensity_d):
    
    output_txt = ('output/' + filename + '/' + filename + '_' + patch_type + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for d, lifetime in enumerate(lifetime_d):
            f.write('%-3d %.4f %.4f\n' % (
                lifetime, length_d[d], intensity_d[d]))

def count_order(patch_sequence_d, filename):
    
    patch_bridge_tail=0
    patch_tail_bridge=0
    bridge_tail_patch=0
    bridge_patch_tail=0
    tail_bridge_patch=0
    tail_patch_bridge=0
    
    patch_tail=0
    bridge_tail=0
    patch_bridge=0
    tail_bridge=0
    bridge_patch=0
    tail_patch=0
    
    segmented_patches_z = separate_segments(patch_sequence_d)
    
    for z, segmented_patches_d in enumerate(segmented_patches_z):
        
        sequence = []
        
        for d, segmented_patches_n in enumerate(segmented_patches_d):
            
            patch_count = Counter(segmented_patches_n).most_common(1)[0][0]
            
            sequence.append(patch_count)
        
        print(sequence)
        
        if len(sequence) > 2:
            
            for n, patch in enumerate(sequence):
                
                if n < (len(sequence) - 2):
                    
                    if patch == 'a' and sequence[n+1] == 'b' and sequence[n+2] == 't':
                        patch_bridge_tail+=1
                    
                    if patch == 'a' and sequence[n+1] == 't' and sequence[n+2] == 'b':
                        patch_tail_bridge+=1
                        
                    if patch == 'b' and sequence[n+1] == 't' and sequence[n+2] == 'a':
                        bridge_tail_patch+=1
                        
                    if patch == 'b' and sequence[n+1] == 'a' and sequence[n+2] == 't':
                        bridge_patch_tail+=1
                        
                    if patch == 't' and sequence[n+1] == 'b' and sequence[n+2] == 'a':
                        tail_bridge_patch+=1
                        
                    if patch == 't' and sequence[n+1] == 'a' and sequence[n+2] == 'b':
                        tail_patch_bridge+=1
        
        if len(sequence) > 1:
            
            for n, patch in enumerate(sequence):
                
                if n < (len(sequence) - 1):
                    
                    if patch == 'a' and sequence[n+1] == 't' :
                        patch_tail+=1
                    
                    if patch == 'b' and sequence[n+1] == 't' :
                        bridge_tail+=1
                        
                    if patch == 'a' and sequence[n+1] == 'b' :
                        patch_bridge+=1
                    
                    if patch == 't' and sequence[n+1] == 'b' :
                        tail_bridge+=1
                        
                    if patch == 'b' and sequence[n+1] == 'a' :
                        bridge_patch+=1
                    
                    if patch == 't' and sequence[n+1] == 'a' :
                        tail_patch+=1                        

    output_txt = ('output/' + filename + '/' + filename + '_sequence_count.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        f.write('%-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d %-3d' % (
            patch_bridge_tail, patch_tail_bridge, bridge_tail_patch, bridge_patch_tail,
            tail_bridge_patch, tail_patch_bridge, patch_tail, bridge_tail, patch_bridge,
            tail_bridge, bridge_patch, tail_patch))


def filter_props_patches(lifetime_cutoff=3, ir_min=1.3):
    
    patch_bridge_tail=0
    patch_tail_bridge=0
    bridge_tail_patch=0
    bridge_patch_tail=0
    tail_bridge_patch=0
    tail_patch_bridge=0
    
    patch_tail=0
    bridge_tail=0
    patch_bridge=0
    tail_bridge=0
    bridge_patch=0
    tail_patch=0
        
        
    for folder_txt in sorted(glob('output/*')):
        
        filename = (folder_txt.split('/')[1])
        print(filename)
        
        for file in sorted(glob(op.join(folder_txt, '*f1*.txt'))):
            
            lifetime_d = []
            max_length_d = []
            max_intensity_ratio_d = []
            
            patch_type = file.split('.txt')[0].split('_')[-1]
            print(patch_type)
            
            with open(file, 'r') as f:
                lines=f.readlines()
                for x in lines:
                    unclean_split = x.split(' ')
                    clean_split = []
                    for item in unclean_split:
                        if len(item) >= 1:
                            clean_split.append(float(item))
                    
                    lifetime_d.append(clean_split[0])
                    max_length_d.append(clean_split[1])
                    max_intensity_ratio_d.append(clean_split[2])
                f.close()
        
            lifetime_a = []
            max_length_a = []
            max_intensity_ratio_a = []
        
            for (lifetime, max_length, max_intensity_ratio) in zip(lifetime_d, max_length_d, max_intensity_ratio_d):
                if (lifetime > lifetime_cutoff) and (max_intensity_ratio > ir_min):
                    lifetime_a.append(lifetime)
                    max_length_a.append(max_length)
                    max_intensity_ratio_a.append(max_intensity_ratio)
                    
            # Convert number of frames to actual lifetime.
            # Imaging interval: 2s
            lifetime_a = [lifetime * 2 for lifetime in lifetime_a]
            
            # Convert number of pixels to micron2. Pixel size: 0.267 um.
            max_length_a = [max_length * 0.16 for max_length in max_length_a]
            
            output_txt = ('output/filtered/' + filename + '_' + patch_type + '.txt')
            
            sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
            with open(output_txt, 'w') as f:
                for a, lifetime in enumerate(lifetime_a):
                    f.write('%-3d  %.4f  %.4f\n' % (
                        lifetime, max_length_a[a], max_intensity_ratio_a[a]))

        for file in sorted(glob(op.join(folder_txt, '*_sequence_count.txt'))):
             
            with open(file, 'r') as f:
                lines=f.readlines()
                for x in lines:
                    unclean_split= x.split(' ')
                    list_of_strings = []
                    for item in unclean_split:
                        if len(item) >= 1:
                            list_of_strings.append(item)
                     
                    print(list_of_strings)
                    clean_split = [int(n) for n in list_of_strings]
                    
                    patch_bridge_tail += clean_split[0]
                    patch_tail_bridge += clean_split[1]
                    bridge_tail_patch += clean_split[2]
                    bridge_patch_tail += clean_split[3]
                    tail_bridge_patch += clean_split[4]
                    tail_patch_bridge += clean_split[5]
                    
                    patch_tail += clean_split[6]
                    bridge_tail += clean_split[7]
                    patch_bridge += clean_split[8]
                    tail_bridge += clean_split[9]
                    bridge_patch += clean_split[10]
                    tail_patch += clean_split[11]
                f.close()

    titles = ['patch-bridge-tail', 'patch-tail-bridge', 'bridge-tail-patch',
              'bridge-patch-tail', 'tail-bridge-patch', 'tail-patch-bridge',
              'patch-tail', 'bridge-tail', 'patch-bridge', 'tail-bridge', 
              'bridge-patch', 'tail-patch']
    
    numbers = [patch_bridge_tail, patch_tail_bridge, bridge_tail_patch, bridge_patch_tail,
               tail_bridge_patch, tail_patch_bridge, patch_tail, bridge_tail, patch_bridge,
               tail_bridge, bridge_patch, tail_patch]
    
    output_txt = ('output/consolidated/sequence_count.txt')
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        f.write("\t".join(titles) + "\n")
        f.write("\t".join(map(str, numbers)))
    
    patch_types = ['bridge', 'tail', 'actin-bound-patch']
    
    for patch_type in patch_types:
        
        lifetime_a = []
        max_length_a = []
        max_intensity_ratio_a = []

        for file in sorted(glob('output/filtered/*_' + patch_type + '.txt')):
            
            print(file)
            
            with open(file, 'r') as f:
                lines=f.readlines()
                for x in lines:
                    unclean_split = x.split(' ')
                    clean_split = []
                    for item in unclean_split:
                        if len(item) >= 1:
                            clean_split.append(float(item))
                    
                    lifetime_a.append(clean_split[0])
                    max_length_a.append(clean_split[1])
                    max_intensity_ratio_a.append(clean_split[2])
                f.close()
        
        print('saving')
        
        output_txt = ('output/consolidated/' + patch_type + '.txt')    
        sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
        with open(output_txt, 'w') as f:
            for a, lifetime in enumerate(lifetime_a):
                f.write('%-3d  %.4f  %.4f\n' % (
                    lifetime, max_length_a[a], max_intensity_ratio_a[a]))
                


if __name__ == '__main__':
    for folder_tif in sorted(glob('data-tiff/*')):
        compute_patches(folder_tif)

    #filter_props_patches() 
