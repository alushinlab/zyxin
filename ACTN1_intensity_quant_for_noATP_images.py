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

def compute_coloc(folder_tif):
    all_actn1_ratios = []
    all_zyx_ratios = []
    num_events = []
    
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif'))):
        
        print('Loading ... ' + tif_filename)
        filename = (tif_filename.split('.')[0]).split('tif')[1]
        
        name, img_yxc = load_TIF_series(tif_filename)
    
        ### make masks
        (actn1_zyx, actn1_nozyx, actn1_mask, zyx_mask) = make_masks(img_yxc)
        
        ### calculate # of zyxin - actn1 binding events
        actn1_zyx_labeled = sme.label(actn1_zyx, connectivity=2)
        actn1_zyx_regprops = sme.regionprops(actn1_zyx_labeled)
        actn1_zyx_events = len(actn1_zyx_regprops)
        
        ### calculate intensity of actn1/zyx inside colocalized regions
        if actn1_zyx_events > 0: 
            actn1_ratio = cal_intR(actn1_zyx_regprops, img_yxc, actn1_mask, channel=1)
            zyxin_ratio = cal_intR(actn1_zyx_regprops, img_yxc,  zyx_mask, channel=2)
            all_actn1_ratios.extend(actn1_ratio)
            all_zyx_ratios.extend(zyxin_ratio)
        
        num_events.append(actn1_zyx_events)
        
    zipped = itertools.zip_longest(num_events, all_actn1_ratios, all_zyx_ratios,
                                    fillvalue=" ")
    
    fn = (folder_tif.split('/')[1]).split('tif')[0]
    output_txt = ('output/' + fn + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    with open(output_txt, 'w') as f:
        for data in zipped:
            print(*data, file=f, sep=" ")
        
def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

''' for actn1 thresholding, use yen for ABM, and li for a WT'''
def make_masks(img_yxc):
    
    ###  generate actin mask
    raw_actin = img_yxc[:, :, 0]
    
    actin_thresh = raw_actin > sf.threshold_li(raw_actin)
    actin_mask = np.array(smo.remove_small_objects(actin_thresh, 5, 2))
    
    #si.imsave('actin.tiff', actin_mask, check_contrast=False)
        
    ### generate zyxin mask
    raw_zyx = img_yxc[:, :, 2]
    
    zyx_filtered = sf.sobel(raw_zyx)
    zyx_thresh = zyx_filtered > sf.threshold_yen(zyx_filtered)
    zyx_filled = sni.binary_fill_holes(zyx_thresh)
    zyx_mask = smo.binary_erosion(zyx_filled)
    zyx_actin = zyx_mask * actin_mask
    zyx_clean = np.array(smo.remove_small_objects((zyx_actin), 3, 2))

    #si.imsave('zyx-mask.tiff', np.array(zyx_clean), check_contrast=False)
    
    ### generate actn1 masks
    
    raw_actn1 = img_yxc[:, :, 1]
    
    actn1_filtered = sf.sobel(raw_actn1)
    '''change thresholding dependending on WT vs ABM ''' 
    actn1_thresh = actn1_filtered > sf.threshold_li(actn1_filtered)
    actn1_filled = sni.binary_fill_holes(actn1_thresh)
    actn1_mask = smo.binary_erosion(actn1_filled)
    actn1_actin = actn1_mask * actin_mask
    actn1_clean = np.array(smo.remove_small_objects((actn1_actin), 3, 2))
    
    #si.imsave('actn1.tiff', np.array(actn1_clean), check_contrast=False)
    
    ### generate +/- zyx actn mask
    
    actn1_zyx = smo.remove_small_objects((zyx_clean * actn1_clean), 4, 2)
    actn1_nozyx = smo.remove_small_objects((~zyx_clean & actn1_clean), 4, 2)
    
    #si.imsave('actn_zyx.tiff', actn1_zyx, check_contrast=False)
    #si.imsave('actn1_nozyx.tiff', actn1_nozyx, check_contrast=False)
    
    return (actn1_zyx, actn1_nozyx, actn1_clean, zyx_clean)
    
def cal_intR(region_props, img_yxc, mask, channel=2):
    
    ny, nx, nc = img_yxc.shape
    intensity_ratio_list = []
    image = img_yxc[:, :, channel]

    for reg in region_props:
        y_i, x_i = reg.coords.T  # split the two columns
        radius = max(1, int(2 * np.sqrt(reg.area)))
    
        inner = np.zeros((ny, nx), dtype=bool)
        inner[y_i, x_i] = 1
        outer = smo.binary_dilation(inner, smo.disk(radius)) & ~inner & ~mask

        inner_mean = image[inner].mean()
        outer_mean = image[outer].mean()
        intensity_ratio = inner_mean / outer_mean
        intensity_ratio_list.append(intensity_ratio)
    
    return(intensity_ratio_list)
    
            
if __name__ == '__main__':
    for folder_tif in sorted(glob('tiff/*')):
        print(' ---- ' + folder_tif + ' --- ')
        compute_coloc(folder_tif)
    
    #for folder_txt in sorted(glob('output/*')):
        #filter_props(folder_txt)
            #count_events(folder_txt)
