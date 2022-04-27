#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:52:58 2021

@author: fang
"""

import numpy as np
import os
import math


def get_file_size(file_path): 
    return os.path.getsize(file_path)




def RGB2YUV(rgb):
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    y = 0.212600 * r + 0.715200 * g + 0.072200 * b
    u = -0.114572 * r - 0.385428 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.454153 * g - 0.045847 * b + 128.0
    
    yuv = np.concatenate((y[:,None],u[:,None],v[:,None]),-1)
    
    return yuv



def YUV2RGB(yuv):
    y1, u1, v1 = yuv[:,0], yuv[:,1]-128, yuv[:,2]-128
    r = y1 + 1.57480 * v1
    g = y1 - 0.18733 * u1 - 0.46813 * v1
    b = y1 + 1.85563 * u1
    
    rgb = np.concatenate((r[:,None],g[:,None],b[:,None]),-1)
    
    return rgb




def get_PSNR(y1, y2):
    
    max_energy = 255*255
    psnr = 10 * math.log10( (max_energy) / np.mean((y1-y2)**2) )
    return psnr


def get_PSNR_yuv(yuv1, yuv2):
    
    psnr_y = get_PSNR(yuv1[:,0], yuv2[:,0])
    psnr_u = get_PSNR(yuv1[:,1], yuv2[:,1])
    psnr_v = get_PSNR(yuv1[:,2], yuv2[:,2])
    
    psnr_yuv = (6*psnr_y + psnr_u + psnr_v)/8
    
    return psnr_yuv


def eval_rec(V, C, CT_q, Qstep, depth, inv_haar3D):
    CT_q = np.round(CT_q)
    CT_q= CT_q*Qstep
    C_rec = inv_haar3D(V, CT_q, depth)
    psnr = get_PSNR(C[:,0], C_rec[:,0])   
    return psnr



    

from torchsparse import SparseTensor
def get_sp_tensor(points):

    center = (points.max(0)[0]+points.min(0)[0])/2
    points_sp = SparseTensor((points-center)[:,:3]/(points.max()+1e-6), points)
    
    return points_sp

def get_sp_tensor_feat(points, feat):
        
    #center = (points.max(0)[0]+points.min(0)[0])/2
    points_sp = SparseTensor(feat, points)
    
    return points_sp



if __name__ == '__main__':
    import scipy.io
    mat = scipy.io.loadmat('./sample_data/scene0000_00_vh_clean_2.mat')
    
    V=mat['V']
    C=mat['C']*255
    
    import pptk
    v=pptk.viewer(V)
    v.attributes(C/255)
    v.set(point_size=0.5)
    
    
    
    yuv = RGB2YUV(C)
    
    
    # import pptk
    # v=pptk.viewer(V)
    # v.attributes(yuv/255)
    # v.set(point_size=0.5)
    
    
    C2 = YUV2RGB(yuv)
    
    import pptk
    v=pptk.viewer(V)
    v.attributes(C2/255)
    v.set(point_size=0.5)
    


    temp=get_PSNR_yuv(RGB2YUV(C), RGB2YUV(C-1))
    print(temp)

    temp=get_PSNR_yuv(RGB2YUV(C), RGB2YUV(C-10))
    print(temp)

    temp=get_PSNR_yuv(RGB2YUV(C), RGB2YUV(C-100))
    print(temp)











