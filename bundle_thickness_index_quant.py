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
        
        filename = (tif_filename.split('.tif')[0]).split('/')[2]
        print('Loading ... ' + filename)
        
        name, raw_tyxc = load_TIF_series(tif_filename)
        
        print('Making masks...')
        (actin_mask_t) = make_masks(raw_tyxc, filename)

        print('Computing and saving...')
        compute_properties(actin_mask_t, raw_tyxc, filename)
        
def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(raw_tyxc, fn, t_start=0, window_size=2):
    
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
    
    
    ###  generate actin mask from averaged movie 
    raw_actin_t = avg_tyxc[:, :, :, 0]
    
    sm_actin_t = sf.median(raw_actin_t)
    actin_thresh = sm_actin_t > sf.threshold_otsu(sm_actin_t)
    actin_mask_t = smo.remove_small_objects(actin_thresh, 50, 2)
    
    output_path = 'output/masks/' + fn + '.tif'
    sh.mkdir('-p', op.dirname(op.abspath(output_path)))
    si.imsave(output_path, actin_mask_t, check_contrast=False)
    
    return(actin_mask_t)

def compute_properties(actin_mask_t, raw_ctyx, fn, channel=0):
    
    # Compute number of erosions necessary to reach ##% of area in first frame
    total_erosion_count = []
    total_ratios = []
    for t, actin_mask in enumerate(actin_mask_t):
        
        # Erosion count as a measure of stress fiber thickness index
        erosion_count = 0
        area_threshold = actin_mask_t[0].sum()
        eroded = actin_mask
        area_prev = eroded.sum()
        while True:
            
            if area_prev <= area_threshold:
                break
            
            eroded = smo.binary_erosion(eroded)
            erosion_count += 1
            
            area_eroded = eroded.sum()
            
            if area_eroded <= area_threshold:
                break
            if area_eroded == area_prev:
                break
        total_erosion_count.append(erosion_count)
        
        # Compute actin intensity ratio inside and outside mask
        raw_actin = raw_ctyx[t, :, :, channel]
        
        actin_masked = actin_mask * raw_actin
        avg_actin = np.average(actin_masked)
        
        bg_mask = (~actin_mask) * raw_actin
        avg_bg = np.average(bg_mask)
        
        ratio = avg_actin / avg_bg
        total_ratios.append(ratio)
    
    output_txt = ('output/' + fn + '.txt')    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    
    with open(output_txt, 'w') as f:
        for t, erosion in enumerate(total_erosion_count):
            f.write('%-3d  %.4f\n' % (erosion, total_ratios[t]))

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
                 
if __name__ == '__main__':
    for folder_tif in sorted(glob('tiff/*')):
        compute_patches(folder_tif)