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

def compute_patches(folder_tif):
    
    folder_nm = folder_tif.split('/')[1]
    
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif*'))):
        
        filename = (tif_filename.split('.tif')[0]).split('/')[2]
        print('Loading ... ' + filename)
        
        name, raw_tyxc = load_TIF_series(tif_filename)
        
        print('Making masks...')
        (actin_mask_t) = make_masks(raw_tyxc, filename, folder_nm)

        print('Computing and saving...')
        compute_properties(actin_mask_t, raw_tyxc, filename, folder_nm)
        
def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(raw_tyxc, fn, folder_nm, t_start=0, window_size=2):
    print(fn)
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
    
    output_path = 'output/masks/' + folder_nm + '/' + fn + '.tif'
    sh.mkdir('-p', op.dirname(op.abspath(output_path)))
    si.imsave(output_path, actin_mask_t, check_contrast=False)
    
    return(actin_mask_t)

def compute_properties(actin_mask_t, raw_ctyx, fn, folder_nm, channel=0):
    
    # Compute number of erosions necessary to reach ##% of area in first frame
    total_den = []
    total_area = []
    for t, actin_mask in enumerate(actin_mask_t):
        # Measure integrated density of frame
        int_den = np.sum(raw_ctyx[t, :, :, 0])
        total_den.append(int_den)
        
        # Measure area of mask each frame
        mask = actin_mask*1
        bg_mask = ~actin_mask * 1
        actin_area = np.sum(mask)
        total_area.append(actin_area)
        
        # normalize to average of first 5 frames
        avg_f5 = np.sum(total_den[0:4])/5 
        avg_area_f5 = np.sum(total_den[0:4])/5 
        norm_den = total_den / avg_f5
        norm_area = total_area / avg_area_f5

    output_txt = ('output/' + folder_nm + '/' + fn + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    
    with open(output_txt, 'w') as f:
        for t, area in enumerate(total_area):
            f.write('%-3d %.4f %.4f %.4f\n' % (area, total_den[t], norm_area[t]
                                               , norm_den[t]))

def filter_props(folder_txt):

    filename = (folder_txt.split('/')[1])
    output_txt = ('output/filtered/' + filename + '.csv')    
    output_txt_avg = ('output/filtered-rollavg-10frames/' + filename + '.csv')    
    all_norm_den = []
    
    for file in sorted(glob(op.join(folder_txt, '*.txt'))):
        
        print('Loading: ' + file)
        
        norm_den = []
        
        with open(file, 'r') as f:
            lines=f.readlines()
            for x in lines:
                unclean_split = x.split(' ')
                clean_split = []
                for item in unclean_split:
                    if len(item) >= 1:
                        clean_split.append(float(item))
                
                norm_den.append(clean_split[3])
            f.close()
        
        all_norm_den.append(norm_den)
    
    all_roll_avgs = []
    # create rolling average 
    for norm_den_list in all_norm_den:
        window_size = 10
  
        i = 0
        # Initialize an empty list to store moving averages
        roll_avgs = []
          
        # Loop through the array t o
        #consider every window of size 3
        while i < len(norm_den_list) - window_size + 1:
          
            # Calculate the average of current window
            window_avg = round(np.sum(norm_den_list[i:i+window_size]) / window_size, 2)
              
            # Store the average of current
            # window in moving average list
            roll_avgs.append(window_avg)
              
            # Shift window to right by one position
            i += 1
        
        all_roll_avgs.append(roll_avgs)
  
    rollavg_rows = itertools.zip_longest(*all_roll_avgs, fillvalue=0)
    norm_den_rows = itertools.zip_longest(*all_norm_den, fillvalue=0)
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt_avg)))
    #with open(output_txt, 'w') as f:
    with open(output_txt_avg, "w") as f:
        writer = csv.writer(f)
        for row in rollavg_rows:
            writer.writerow(row)
    
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, "w") as f:
        writer = csv.writer(f)
        for row in norm_den_rows:
            writer.writerow(row)
                 
if __name__ == '__main__':
    #for folder_tif in sorted(glob('tiff_20/*')):
        #compute_patches(folder_tif)
    for folder in sorted(glob('output/*')):
        filter_props(folder)