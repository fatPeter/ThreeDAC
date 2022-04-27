#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:36:28 2022

@author: fang
"""

import numpy as np
import torch
from Haar3D_inform import haar3D, inv_haar3D
import utils



from torchsparse import SparseTensor
def get_sp_tensor(points):

    center = (points.max(0)[0]+points.min(0)[0])/2
    points_sp = SparseTensor((points-center)[:,:3]/(points.max()+1e-6), points)
    
    return points_sp

def get_sp_tensor_feat(points, feat):
        
    #center = (points.max(0)[0]+points.min(0)[0])/2
    points_sp = SparseTensor(feat, points)
    
    return points_sp

    
        


# ScanNet

def InitialCoding_ScanNet(points, colors, depth, Qstep):
    
    # RAHT
    res = haar3D(points, colors, depth)
    
    # process side information
    CT, depth_CT, w, node_xyz = res['CT'], res['depth_CT'], res['w'], res['node_xyz']
    C_rec = inv_haar3D(points, np.round(CT/Qstep)*Qstep, depth)
    res_rec = haar3D(points, C_rec, depth)   

  
    
    
    low_freq = res_rec['low_freq']         
    low_freq[low_freq<=0]=0.1    
    
    cyuv = low_freq/np.sqrt(w[:,None])
    crgb = utils.YUV2RGB(cyuv)
    
    
    iCT_low = res_rec['iCT_low']
    for i in range(len(iCT_low)):
        iCT_low[i][iCT_low[i]<=0]=0.1 
        

    iW, iPos = res['iW'], res['iPos']
    
    iyuv, irgb =[], []
    for i in range(len(iCT_low)):
        yuv = iCT_low[i]/np.sqrt(iW[i][:,None])
        rgb = utils.YUV2RGB(yuv)
        iyuv.append(yuv)         
        irgb.append(rgb)         
    
    
    for i in range(len(iW)):
        iW[i]=torch.tensor(iW[i])
        iPos[i]=torch.tensor(iPos[i])
        iCT_low[i]=torch.tensor(iCT_low[i])
        iyuv[i]=torch.tensor(iyuv[i])
        irgb[i]=torch.tensor(irgb[i])
        
        
        
                
    
    
    CT, w, depth_CT, node_xyz = torch.tensor(CT), torch.tensor(w), torch.tensor(depth_CT), torch.tensor(node_xyz)
    low_freq, cyuv, crgb = torch.tensor(low_freq), torch.tensor(cyuv), torch.tensor(crgb)
    
    
    
    # align tree node position to point cloud space
    temp=[[1,1,2],[1,2,2],[2,2,2]]
    factor_list=[]
    for i in range(depth):
        for j in range(3):
            factor = np.array(temp[j])*(2**i)
            factor_list.append(factor)
            
    shift_list=[]
    for factor in factor_list:
        shift = (2**(np.log2(factor))-1)/2
        shift_list.append(shift)    
        
        
    level_list = [0]
    node_xyz_resize = torch.zeros(node_xyz.shape)
    for i in range(depth*3):
        node_xyz_resize[depth_CT==i] = (node_xyz[depth_CT==i]*factor_list[i]+shift_list[i].astype(int)).float()
        
        level_list.append(torch.sum(depth_CT==i)+level_list[-1])
        
    node_xyz_resize[0]=0
        
    for i in range(depth*3):
        iPos[i] = iPos[i]*factor_list[i]+shift_list[i].astype(int)        
    
    
    
    
    res = {'CT':CT, 
           'w':w, 
           'depth_CT':depth_CT, 
           'node_xyz':node_xyz_resize,
           'low_freq':low_freq,
           'cyuv':cyuv,
           'crgb':crgb,
           
           'iCT_low':iCT_low,
           'iW':iW,
           'iPos':iPos,
           'iyuv':iyuv,
           'irgb':irgb,
           'level_list':level_list
           }
    
    return res
        

    
    
    # import pptk
    
    # temp_pc = np.concatenate(iPos,0)
    
    
    # v=pptk.viewer(temp_pc)
    # v.set(point_size=1)    
    
    
    




def Construct_InitialCoding_Info_ScanNet(s_info, Qstep, depth):
    
    

    CT = s_info['CT']
    low_freq, cyuv, crgb = s_info['low_freq'], s_info['cyuv'], s_info['crgb']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv, irgb = s_info['iyuv'], s_info['irgb']
    
    

 
    

    
    # construct tensor and sparse tensor
    
    # construct tensor
    high_freq_info = []
    
    
    high_freq_info_t = torch.cat((depth_CT[:,None], 
                          torch.log(w[:,None]), 
                          torch.log(low_freq), 
                          cyuv/255, 
                          crgb/255,
                          # node_xyz/(2**depth-1)
                          ),-1).float()    
    

    # query points
    high_freq_nodes = []
    
   
     
    
    
    # tensor for initial coding context
    low_freq_nodes = []
    low_freq_feat = []
    


    for i in range(depth*3):
        mask = depth_CT==i
        high_freq_info.append(high_freq_info_t[mask])
        
        
        
        temp = torch.tensor(node_xyz[depth_CT==i])
        temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
        high_freq_nodes.append(temp)        
        
        
        temp = torch.tensor(iPos[i])
        temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
        low_freq_nodes.append(temp)
  
        
        
        depth_temp = torch.zeros(iPos[i].shape[0])+i
        weight_temp = torch.log(iW[i].float())
        low_f_temp = torch.log(iCT_low[i])
        
        cyuv_temp = iyuv[i]
        crgb_temp = irgb[i]
        cyuv_temp = torch.tensor(cyuv_temp/255)
        crgb_temp = torch.tensor(crgb_temp/255)
        
        
        pos_temp = torch.tensor(iPos[i]/(2**depth-1))
        
        #feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp, pos_temp), -1)
        feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp), -1)
        low_freq_feat.append(feat_temp)
    
        
    
    high_freq_info = torch.cat(high_freq_info,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    low_freq_nodes = torch.cat(low_freq_nodes,0)
    low_freq_feat = torch.cat(low_freq_feat,0)
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    
    low_freq_nodes_sp = get_sp_tensor_feat(low_freq_nodes.long(), low_freq_feat.float())

  
    res = {'high_freq_info':high_freq_info, 
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'low_freq_nodes_sp':low_freq_nodes_sp, 
            }
    
    return res



def Construct_InterChannel_Info_ScanNet(s_info, Qstep, depth):
    
    

    CT = s_info['CT']
    low_freq, cyuv, crgb = s_info['low_freq'], s_info['cyuv'], s_info['crgb']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv, irgb = s_info['iyuv'], s_info['irgb']
    
    
    # construct tensor
    inter_channel_info_y = []
    inter_channel_info_yu = []
    
    input_y = torch.round(CT/Qstep)[:,0:1]
    input_yu = torch.round(CT/Qstep)[:,0:2]    
    inter_channel_info_y_t = input_y.float()
    inter_channel_info_yu_t = input_yu.float()
    

    
    # construct tensor and sparse tensor

    # query points
    high_freq_nodes = []
    
    # # construct sparse tensor
    # temp = torch.tensor(node_xyz)
    # temp = torch.cat((temp, depth_CT[:,None]),-1)            
    # high_freq_nodes.append(temp)      
    
    
    # tensor for inter channel context
    high_freq_y = []
    high_freq_yu = []

    # # construct sparse tensor
    # temp = np.round(CT/Qstep)
    # high_freq_y.append(temp[:,0:1])
    # high_freq_yu.append(temp[:,0:2])
    
    for i in range(depth*3):
        mask = depth_CT==i
        inter_channel_info_y.append(inter_channel_info_y_t[mask])
        inter_channel_info_yu.append(inter_channel_info_yu_t[mask])
        
        
        
        temp = torch.tensor(node_xyz[depth_CT==i])
        temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
        high_freq_nodes.append(temp)
        
        
        temp = np.round(CT[depth_CT==i]/Qstep)
        high_freq_y.append(temp[:,0:1])
        high_freq_yu.append(temp[:,0:2])
                    
       
        
    
    
    inter_channel_info_y = torch.cat(inter_channel_info_y,0)
    inter_channel_info_yu = torch.cat(inter_channel_info_yu,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    high_freq_y = torch.cat(high_freq_y,0)
    high_freq_yu = torch.cat(high_freq_yu,0)
    
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    high_freq_nodes_y_sp = get_sp_tensor_feat(high_freq_nodes.long(), high_freq_y.float())
    high_freq_nodes_yu_sp = get_sp_tensor_feat(high_freq_nodes.long(), high_freq_yu.float())
    
  
    res = {
            'inter_channel_info_y':inter_channel_info_y, 
            'inter_channel_info_yu':inter_channel_info_yu,         
        
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'high_freq_nodes_y_sp':high_freq_nodes_y_sp, 
            'high_freq_nodes_yu_sp':high_freq_nodes_yu_sp,
            }
    
    return res




def Construct_InitialCoding_Info_ScanNet_depth(s_info, Qstep, depth_idx):
    
    

    CT = s_info['CT']
    low_freq, cyuv, crgb = s_info['low_freq'], s_info['cyuv'], s_info['crgb']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv, irgb = s_info['iyuv'], s_info['irgb']
    
    

 
    

    
    # construct tensor and sparse tensor
    
    # construct tensor
    high_freq_info = []
    
    
    high_freq_info_t = torch.cat((depth_CT[:,None], 
                          torch.log(w[:,None]), 
                          torch.log(low_freq), 
                          cyuv/255, 
                          crgb/255,
                          # node_xyz/(2**depth-1)
                          ),-1).float()    
    

    # query points
    high_freq_nodes = []
    
   
     
    
    
    # tensor for initial coding context
    low_freq_nodes = []
    low_freq_feat = []
    


    i = depth_idx
    mask = depth_CT==i
    high_freq_info.append(high_freq_info_t[mask])
    
    
    
    temp = torch.tensor(node_xyz[depth_CT==i])
    temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
    high_freq_nodes.append(temp)        
    
    
    temp = torch.tensor(iPos[i])
    temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
    low_freq_nodes.append(temp)
  
    
    
    depth_temp = torch.zeros(iPos[i].shape[0])+i
    weight_temp = torch.log(iW[i].float())
    low_f_temp = torch.log(iCT_low[i])
    
    cyuv_temp = iyuv[i]
    crgb_temp = irgb[i]
    cyuv_temp = torch.tensor(cyuv_temp/255)
    crgb_temp = torch.tensor(crgb_temp/255)
    
    
    
    #feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp, pos_temp), -1)
    feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp), -1)
    low_freq_feat.append(feat_temp)

        
    
    high_freq_info = torch.cat(high_freq_info,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    low_freq_nodes = torch.cat(low_freq_nodes,0)
    low_freq_feat = torch.cat(low_freq_feat,0)
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    
    low_freq_nodes_sp = get_sp_tensor_feat(low_freq_nodes.long(), low_freq_feat.float())

  
    res = {'high_freq_info':high_freq_info, 
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'low_freq_nodes_sp':low_freq_nodes_sp, 
            }
    
    return res



def Construct_InterChannel_Info_ScanNet_depth(s_info, Qstep, depth_idx):
    
    

    CT = s_info['CT']
    low_freq, cyuv, crgb = s_info['low_freq'], s_info['cyuv'], s_info['crgb']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv, irgb = s_info['iyuv'], s_info['irgb']
    
    
    # construct tensor
    inter_channel_info_y = []
    inter_channel_info_yu = []
    
    input_y = torch.round(CT/Qstep)[:,0:1]
    input_yu = torch.round(CT/Qstep)[:,0:2]    
    inter_channel_info_y_t = input_y.float()
    inter_channel_info_yu_t = input_yu.float()
    

    
    # construct tensor and sparse tensor

    # query points
    high_freq_nodes = []
    
    # # construct sparse tensor
    # temp = torch.tensor(node_xyz)
    # temp = torch.cat((temp, depth_CT[:,None]),-1)            
    # high_freq_nodes.append(temp)      
    
    
    # tensor for inter channel context
    high_freq_y = []
    high_freq_yu = []

    # # construct sparse tensor
    # temp = np.round(CT/Qstep)
    # high_freq_y.append(temp[:,0:1])
    # high_freq_yu.append(temp[:,0:2])
    
    
    i = depth_idx
    mask = depth_CT==i
    inter_channel_info_y.append(inter_channel_info_y_t[mask])
    inter_channel_info_yu.append(inter_channel_info_yu_t[mask])
    
    
    
    temp = torch.tensor(node_xyz[depth_CT==i])
    temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
    high_freq_nodes.append(temp)
    
    
    temp = np.round(CT[depth_CT==i]/Qstep)
    # query_points_featy_temp.append(temp[:,0:1]/torch.sqrt(w[depth_CT==i,None]))
    # query_points_featyu_temp.append(temp[:,0:2]/torch.sqrt(w[depth_CT==i,None]))
    high_freq_y.append(temp[:,0:1])
    high_freq_yu.append(temp[:,0:2])
                
       
        
    
    
    inter_channel_info_y = torch.cat(inter_channel_info_y,0)
    inter_channel_info_yu = torch.cat(inter_channel_info_yu,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    high_freq_y = torch.cat(high_freq_y,0)
    high_freq_yu = torch.cat(high_freq_yu,0)
    
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    high_freq_nodes_y_sp = get_sp_tensor_feat(high_freq_nodes.long(), high_freq_y.float())
    high_freq_nodes_yu_sp = get_sp_tensor_feat(high_freq_nodes.long(), high_freq_yu.float())
    
  
    res = {
            'inter_channel_info_y':inter_channel_info_y, 
            'inter_channel_info_yu':inter_channel_info_yu,         
        
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'high_freq_nodes_y_sp':high_freq_nodes_y_sp, 
            'high_freq_nodes_yu_sp':high_freq_nodes_yu_sp,
            }
    
    return res




    
    
# Semantic Kitti
    
    

def InitialCoding_SemanticKitti(points, colors, depth, Qstep):
    
    # RAHT
    res = haar3D(points, colors, depth)
    
    # process side information
    CT, depth_CT, w, node_xyz = res['CT'], res['depth_CT'], res['w'], res['node_xyz']
    C_rec = inv_haar3D(points, np.round(CT/Qstep)*Qstep, depth)
    res_rec = haar3D(points, C_rec, depth)   

  
    
    
    low_freq = res_rec['low_freq']         
    low_freq[low_freq<=0]=0.1    
    
    cyuv = low_freq/np.sqrt(w[:,None])
    
    
    iCT_low = res_rec['iCT_low']
    for i in range(len(iCT_low)):
        iCT_low[i][iCT_low[i]<=0]=0.1 
        

    iW, iPos = res['iW'], res['iPos']
    
    iyuv = []
    for i in range(len(iCT_low)):
        yuv = iCT_low[i]/np.sqrt(iW[i][:,None])
        iyuv.append(yuv)         
    
    
    for i in range(len(iW)):
        iW[i]=torch.tensor(iW[i])
        iPos[i]=torch.tensor(iPos[i])
        iCT_low[i]=torch.tensor(iCT_low[i])
        iyuv[i]=torch.tensor(iyuv[i])
        
        
        
                
    
    
    CT, w, depth_CT, node_xyz = torch.tensor(CT), torch.tensor(w), torch.tensor(depth_CT), torch.tensor(node_xyz)
    low_freq, cyuv = torch.tensor(low_freq), torch.tensor(cyuv)
    
    
    
    # align tree node position to point cloud space
    temp=[[1,1,2],[1,2,2],[2,2,2]]
    factor_list=[]
    for i in range(depth):
        for j in range(3):
            factor = np.array(temp[j])*(2**i)
            factor_list.append(factor)
            
    shift_list=[]
    for factor in factor_list:
        shift = (2**(np.log2(factor))-1)/2
        shift_list.append(shift)    
        
        
    level_list = [0]
    node_xyz_resize = torch.zeros(node_xyz.shape)
    for i in range(depth*3):
        node_xyz_resize[depth_CT==i] = (node_xyz[depth_CT==i]*factor_list[i]+shift_list[i].astype(int)).float()
        
        level_list.append(torch.sum(depth_CT==i)+level_list[-1])
        
    node_xyz_resize[0]=0
        
    for i in range(depth*3):
        iPos[i] = iPos[i]*factor_list[i]+shift_list[i].astype(int)        
    
    
    
    
    res = {'CT':CT, 
           'w':w, 
           'depth_CT':depth_CT, 
           'node_xyz':node_xyz_resize,
           'low_freq':low_freq,
           'cyuv':cyuv,
           
           'iCT_low':iCT_low,
           'iW':iW,
           'iPos':iPos,
           'iyuv':iyuv,
           'level_list':level_list
           }
    
    return res    
    




def Construct_InitialCoding_Info_SemanticKitti(s_info, Qstep, depth):
    
    

    CT = s_info['CT']
    low_freq, cyuv = s_info['low_freq'], s_info['cyuv']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv = s_info['iyuv']
    
    

 
    

    
    # construct tensor and sparse tensor
    
    # construct tensor
    high_freq_info = []
    
    
    high_freq_info_t = torch.cat((depth_CT[:,None], 
                          torch.log(w[:,None]), 
                          torch.log(low_freq), 
                          cyuv/255, 
                          ),-1).float()    
    

    # query points
    high_freq_nodes = []
    
   
     
    
    
    # tensor for initial coding context
    low_freq_nodes = []
    low_freq_feat = []
    


    for i in range(depth*3):
        mask = depth_CT==i
        high_freq_info.append(high_freq_info_t[mask])
        
        
        
        temp = torch.tensor(node_xyz[depth_CT==i])
        temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
        high_freq_nodes.append(temp)        
        
        
        temp = torch.tensor(iPos[i])
        temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
        low_freq_nodes.append(temp)
  
        
        
        depth_temp = torch.zeros(iPos[i].shape[0])+i
        weight_temp = torch.log(iW[i].float())
        low_f_temp = torch.log(iCT_low[i])
        
        cyuv_temp = iyuv[i]
        cyuv_temp = torch.tensor(cyuv_temp/255)
        
        
        pos_temp = torch.tensor(iPos[i]/(2**depth-1))
        
        #feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp, pos_temp), -1)
        feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp), -1)
        low_freq_feat.append(feat_temp)
    
        
    
    high_freq_info = torch.cat(high_freq_info,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    low_freq_nodes = torch.cat(low_freq_nodes,0)
    low_freq_feat = torch.cat(low_freq_feat,0)
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    
    low_freq_nodes_sp = get_sp_tensor_feat(low_freq_nodes.long(), low_freq_feat.float())

  
    res = {'high_freq_info':high_freq_info, 
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'low_freq_nodes_sp':low_freq_nodes_sp, 
            }
    
    return res



def Construct_InitialCoding_Info_SemanticKitti_depth(s_info, Qstep, depth_idx):
    
    

    CT = s_info['CT']
    low_freq, cyuv = s_info['low_freq'], s_info['cyuv']
    depth_CT, w, node_xyz = s_info['depth_CT'], s_info['w'], s_info['node_xyz']
    iCT_low, iW, iPos = s_info['iCT_low'], s_info['iW'], s_info['iPos']
    iyuv = s_info['iyuv']
    
    

 
    

    
    # construct tensor and sparse tensor
    
    # construct tensor
    high_freq_info = []
    
    
    high_freq_info_t = torch.cat((depth_CT[:,None], 
                          torch.log(w[:,None]), 
                          torch.log(low_freq), 
                          cyuv/255, 
                          ),-1).float()    
    

    # query points
    high_freq_nodes = []
    
   
     
    
    
    # tensor for initial coding context
    low_freq_nodes = []
    low_freq_feat = []
    


    i = depth_idx
    mask = depth_CT==i
    high_freq_info.append(high_freq_info_t[mask])
    
    
    
    temp = torch.tensor(node_xyz[depth_CT==i])
    temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
    high_freq_nodes.append(temp)        
    
    
    temp = torch.tensor(iPos[i])
    temp = torch.cat((temp, torch.zeros(temp.shape[0],1)+i),-1)            
    low_freq_nodes.append(temp)
  
    
    
    depth_temp = torch.zeros(iPos[i].shape[0])+i
    weight_temp = torch.log(iW[i].float())
    low_f_temp = torch.log(iCT_low[i])
    
    cyuv_temp = iyuv[i]
    cyuv_temp = torch.tensor(cyuv_temp/255)
    
    
    
    #feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp, crgb_temp, pos_temp), -1)
    feat_temp = torch.cat((depth_temp[:,None], weight_temp[:,None], low_f_temp, cyuv_temp), -1)
    low_freq_feat.append(feat_temp)

        
    
    high_freq_info = torch.cat(high_freq_info,0)
    
    high_freq_nodes = torch.cat(high_freq_nodes,0)
    low_freq_nodes = torch.cat(low_freq_nodes,0)
    low_freq_feat = torch.cat(low_freq_feat,0)
    
    
    high_freq_nodes_sp = get_sp_tensor(high_freq_nodes.long())
    
    low_freq_nodes_sp = get_sp_tensor_feat(low_freq_nodes.long(), low_freq_feat.float())

  
    res = {'high_freq_info':high_freq_info, 
            'high_freq_nodes_sp':high_freq_nodes_sp, 
            'low_freq_nodes_sp':low_freq_nodes_sp, 
            }
    
    return res



