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
import skimage.restoration as sre
import scipy.ndimage as sni
import pickle
import itertools
from tifffile import imwrite

def compute_patches(folder_tif):
    
    bg = 102
    
    folder_nm = folder_tif.split('/')[1]
    
    for tif_filename in sorted(glob(op.join(folder_tif, '*.tif*'))):
        
        filename = (tif_filename.split('.tif')[0]).split('/')[2]
        print('Loading ... ' + tif_filename)
        
        name, raw_tyxc = load_TIF_series(tif_filename)
        
        print('Making masks...')
        (masks, mask_all, img_shape) = make_masks(raw_tyxc, bg, filename, folder_nm)
        
        print('Computing total area...')
        measure_area(mask_all, filename, folder_nm, img_shape)
        
        print('Computing droplet properties...')
            
        if img_shape == 4:
            compute_properties_4(masks, mask_all, raw_tyxc, filename, folder_nm, bg, 0)
                
            #if np.sum(masks[0, :, :, 1]) < 560000:
                #compute_properties_4(masks, raw_tyxc, filename, folder_nm, bg, 1)

            #compute_properties_4(masks, raw_tyxc, filename, folder_nm, bg, 2)


def load_TIF_series(filename):
    raw_tyxc = si.imread(filename)
    name = op.splitext(op.basename(filename))[0]
    return name, raw_tyxc

def make_masks(raw_tyxc, bg, fn, folder_nm):
    print(fn)
    
    img_shape = len(raw_tyxc.shape)
    print(raw_tyxc.shape)
    
    if img_shape == 4:
        nt, ny, nx, nc = raw_tyxc.shape
        raw_488 = sf.gaussian(raw_tyxc[:, :, :, 0])   
        raw_561 = sf.gaussian(raw_tyxc[:, :, :, 1])     
        raw_640 = sf.gaussian(raw_tyxc[:, :, :, 2])
    
    #print(raw_tyxc.shape)
   
    ###  generate mask
    thresh488 = raw_488 > sf.threshold_otsu(raw_488)
    mask488 = smo.remove_small_objects(thresh488, 20, 2)
    
    thresh561 = raw_561 > sf.threshold_otsu(raw_561)
    mask561 = smo.remove_small_objects(thresh561, 20, 2)
    
    thresh640 = raw_640 > sf.threshold_otsu(raw_640)
    mask640 = smo.remove_small_objects(thresh640, 20, 2)
    
    output_path = 'output_area/masks/' + folder_nm + '/' + fn + '-488.tif'
    sh.mkdir('-p', op.dirname(op.abspath(output_path)))
    si.imsave(output_path,np.array(mask488))
    
    output_path = 'output_area/masks/' + folder_nm + '/' + fn + '-561.tif'
    sh.mkdir('-p', op.dirname(op.abspath(output_path)))
    si.imsave(output_path,np.array(mask561))
    
    output_path = 'output_area/masks/' + folder_nm + '/' + fn + '-640.tif'
    sh.mkdir('-p', op.dirname(op.abspath(output_path)))
    si.imsave(output_path,np.array(mask640))
    
    masks = np.stack((mask488, mask561, mask640), axis=-1)
    mask_all = mask488 & mask561 & mask640

    return(masks, mask_all, img_shape)
            
def compute_properties_4(masks, mask_all, raw_ctyx, fn, folder_nm, bg, channel):
    
    nt, ny, nx, nc = raw_ctyx.shape
    
    labeled = [sme.label(mask, connectivity=1) for t, mask in enumerate(mask_all)]
    
    intensity_ratio_n = []
    diameter_n = []
    droplet_area_n = []

    for t, labels in enumerate(labeled):
        
        for reg in sme.regionprops(labels):
            y_i, x_i = reg.coords.T  # split the two columns
            radius = max(1, int(np.sqrt(reg.area/3.14)))
            diameter = radius*2
            
            diameter_n.append(diameter)
            droplet_area_n.append(reg.area)
            print(diameter, reg.area)

    output_txt = ('output_area/' + folder_nm + '/' + fn + '-ch' + str(channel) + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    
    with open(output_txt, 'w') as f:
        for n, diameter in enumerate(diameter_n):
            f.write('%-3d %-3d\n' % (diameter, droplet_area_n[n]))
                 
def measure_area(mask, fn, folder_nm, img_shape):
    
    print(mask.shape)
    
    if img_shape == 4: 
        covered_area_n = []
        
        for t, frame in enumerate(mask):
            mask_x = np.size(frame, 0)
            mask_y = np.size(frame, 1)
            
            total_area = mask_x * mask_y
            masked_area = np.sum(frame)
            covered_area = masked_area / total_area 
            
            print(total_area, masked_area, covered_area)
            covered_area_n.append(covered_area)
            
    if img_shape == 3:
        mask_x = np.size(mask, 0)
        mask_y = np.size(mask, 1)
        
        total_area = mask_x * mask_y
        masked_area = np.sum(mask)
        covered_area = masked_area / total_area 
        
        covered_area_n.append(covered_area)
    
        print(total_area, masked_area, covered_area)
    
    output_txt = ('output_area/' + folder_nm + '/' + fn + '-covered_area' + '.txt')
    sh.mkdir('-p', op.dirname(op.abspath(output_txt)))
    
    with open(output_txt, 'w') as f:
        for n, area in enumerate(covered_area_n):
            f.write('%.9f\n' % (area))
    
if __name__ == '__main__':
    for folder_tif in sorted(glob('area-tiff/*')):
        compute_patches(folder_tif)