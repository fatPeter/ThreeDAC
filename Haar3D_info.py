#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:27:39 2021

@author: fang

It is an python version of RAHT based on https://github.com/digitalivp/RAHT/tree/reorder.
The original C implementation is more readable.

"""

import numpy as np
import torch




 
# morton coding
# convert voxlized and deduplicated point cloud to morton code
def copyAsort(V):
    # input
    # V: np.array (n,3), input vertices
    
    # output
    # W: np.array (n,), weight
    # val: np.array (n,), zyx val of vertices
    # reord: np.array (n,), idx ord after sort
    
    
    
    V=V.astype(np.int64)  
    
    # w of leaf node sets to 1
    W=np.ones(V.shape[0])  
    
    # encode zyx (pos) to bin
    vx, vy, vz= V[:,2], V[:,1], V[:,0]
    val = ((0x000001 & vx)    ) + ((0x000001 & vy)<< 1) + ((0x000001 &  vz)<< 2) + \
                ((0x000002 & vx)<< 2) + ((0x000002 & vy)<< 3) + ((0x000002 &  vz)<< 4) + \
                ((0x000004 & vx)<< 4) + ((0x000004 & vy)<< 5) + ((0x000004 &  vz)<< 6) + \
                ((0x000008 & vx)<< 6) + ((0x000008 & vy)<< 7) + ((0x000008 &  vz)<< 8) + \
                ((0x000010 & vx)<< 8) + ((0x000010 & vy)<< 9) + ((0x000010 &  vz)<<10) + \
                ((0x000020 & vx)<<10) + ((0x000020 & vy)<<11) + ((0x000020 &  vz)<<12) + \
                ((0x000040 & vx)<<12) + ((0x000040 & vy)<<13) + ((0x000040 &  vz)<<14) + \
                ((0x000080 & vx)<<14) + ((0x000080 & vy)<<15) + ((0x000080 &  vz)<<16) + \
                ((0x000100 & vx)<<16) + ((0x000100 & vy)<<17) + ((0x000100 &  vz)<<18) + \
                ((0x000200 & vx)<<18) + ((0x000200 & vy)<<19) + ((0x000200 &  vz)<<20) + \
                ((0x000400 & vx)<<20) + ((0x000400 & vy)<<21) + ((0x000400 &  vz)<<22) + \
                ((0x000800 & vx)<<22) + ((0x000800 & vy)<<23) + ((0x000800 &  vz)<<24) + \
                ((0x001000 & vx)<<24) + ((0x001000 & vy)<<25) + ((0x001000 &  vz)<<26) + \
                ((0x002000 & vx)<<26) + ((0x002000 & vy)<<27) + ((0x002000 &  vz)<<28) + \
                ((0x004000 & vx)<<28) + ((0x004000 & vy)<<29) + ((0x004000 &  vz)<<30) + \
                ((0x008000 & vx)<<30) + ((0x008000 & vy)<<31) + ((0x008000 &  vz)<<32) + \
                ((0x010000 & vx)<<32) + ((0x010000 & vy)<<33) + ((0x010000 &  vz)<<34) + \
                ((0x020000 & vx)<<34) + ((0x020000 & vy)<<35) + ((0x020000 &  vz)<<36) + \
                ((0x040000 & vx)<<36) + ((0x040000 & vy)<<37) + ((0x040000 &  vz)<<38) + \
                ((0x080000 & vx)<<38) + ((0x080000 & vy)<<39) + ((0x080000 &  vz)<<40) + \
                ((0x100000 & vx)<<40) + ((0x100000 & vy)<<41) + ((0x100000 &  vz)<<42)
        
    reord=np.argsort(val)
    val=np.sort(val)
    
    return W, val, reord



# morton decoding
# convert morton code to point cloud
def val2V(val, factor):
    '''

    Parameters
    ----------
    val : morton code
    factor : shift morton code for deocoding

    Returns
    -------
    V_re : point cloud

    '''
    
    if factor>2 or factor<0:
        print('error')
        return
    
    val = val<<factor    
    V_re = np.zeros((val.shape[0],3))
    
    V_re[:,2] = (0x000001 & val) + \
                (0x000002 & (val>> 2)) + \
                (0x000004 & (val>> 4)) + \
                (0x000008 & (val>> 6)) + \
                (0x000010 & (val>> 8)) + \
                (0x000020 & (val>>10)) + \
                (0x000040 & (val>>12)) + \
                (0x000080 & (val>>14)) + \
                (0x000100 & (val>>16)) + \
                (0x000200 & (val>>18)) + \
                (0x000400 & (val>>20)) + \
                (0x000800 & (val>>22)) + \
                (0x001000 & (val>>24)) + \
                (0x002000 & (val>>26)) + \
                (0x004000 & (val>>28)) + \
                (0x008000 & (val>>30)) + \
                (0x010000 & (val>>32)) + \
                (0x020000 & (val>>34)) + \
                (0x040000 & (val>>36)) + \
                (0x080000 & (val>>38)) + \
                (0x100000 & (val>>40))
    
    
    V_re[:,1] = (0x000001 & (val>> 1)) + \
                (0x000002 & (val>> 3)) + \
                (0x000004 & (val>> 5)) + \
                (0x000008 & (val>> 7)) + \
                (0x000010 & (val>> 9)) + \
                (0x000020 & (val>>11)) + \
                (0x000040 & (val>>13)) + \
                (0x000080 & (val>>15)) + \
                (0x000100 & (val>>17)) + \
                (0x000200 & (val>>19)) + \
                (0x000400 & (val>>21)) + \
                (0x000800 & (val>>23)) + \
                (0x001000 & (val>>25)) + \
                (0x002000 & (val>>27)) + \
                (0x004000 & (val>>29)) + \
                (0x008000 & (val>>31)) + \
                (0x010000 & (val>>33)) + \
                (0x020000 & (val>>35)) + \
                (0x040000 & (val>>37)) + \
                (0x080000 & (val>>39)) + \
                (0x100000 & (val>>41))
    
    
    V_re[:,0] = (0x000001 & (val>> 2)) + \
                (0x000002 & (val>> 4)) + \
                (0x000004 & (val>> 6)) + \
                (0x000008 & (val>> 8)) + \
                (0x000010 & (val>>10)) + \
                (0x000020 & (val>>12)) + \
                (0x000040 & (val>>14)) + \
                (0x000080 & (val>>16)) + \
                (0x000100 & (val>>18)) + \
                (0x000200 & (val>>20)) + \
                (0x000400 & (val>>22)) + \
                (0x000800 & (val>>24)) + \
                (0x001000 & (val>>26)) + \
                (0x002000 & (val>>28)) + \
                (0x004000 & (val>>30)) + \
                (0x008000 & (val>>32)) + \
                (0x010000 & (val>>34)) + \
                (0x020000 & (val>>36)) + \
                (0x040000 & (val>>38)) + \
                (0x080000 & (val>>40)) + \
                (0x100000 & (val>>42))
                
    if factor == 1:
        V_re[:,2]/=2
    if factor == 2:
        V_re[:,1]/=2
        V_re[:,2]/=2
    
                
    return V_re












def transform_batched(a0, a1, C0, C1):  
    # input
    # a0, a1: float, weight
    # C0, C1: np.array (n,), att of vertices
    
    # output
    # v0, v1: np.array (n,), trans att of vertices
    
    trans_matrix=np.array([[a0, a1],
                           [-a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))
    
    
    V=np.matmul(trans_matrix, np.concatenate((C0,C1),1))
    
    return V[:,0], V[:,1]


    
    
    
    
def itransform_batched(a0, a1, CT0, CT1):  
    # input
    # a0, a1: float, weight
    # CT0, CT1: np.array (n,), trans att of vertices
    
    # output
    # c0, c1: np.array (n,), att of vertices
    
    trans_matrix=np.array([[a0, -a1],
                           [a1, a0]])
    trans_matrix=trans_matrix.transpose((2,0,1))
    
    C=np.matmul(trans_matrix, np.concatenate((CT0,CT1),1))
    
    return C[:,0], C[:,1]  
    
    
    
    
    


def haar3D(inV, inC, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inC : attributes
    depth : depth level of geometry(octree)

    Returns
    -------
    res : transformed coefficients and side information

    '''
    
    
    import copy
    inC = copy.deepcopy(inC)
    
    
    # N,NN number of points
    # K, dims (3) of geometry
    N, K = inC.shape
    NN = N
    
    # depth of RAHT tree (without leaf node level)
    depth *= 3
    
    # low_freq coeffs for transmitting coeffs (high_freq)
    low_freq = np.zeros(inC.shape)
    
    
    
    
    wT = np.zeros((N, ))
    valT = np.zeros((N, ))
    posT = np.zeros((N, )).astype(np.int64)
    
    
    
    # position of coeffs
    node_xyz = np.zeros((N, 3))-1
    
    
    
    depth_CT = np.zeros((N, ))-1
    
    
    
    
    
    # morton coding
    # return weight, morton code, map from inV to val
    w, val, TMP = copyAsort(inV)
    
    
    
    # pos, order from transformed coeffes to morton sorted attributes
    pos = np.arange(N)
    C = inC[TMP].astype(np.float64)
    
    
    
    # low_freq for each depth
    iCT_low=[]
    # parent idx for each depth
    iparent=[]
    # weight for each depth
    iW=[]
    # node position for each depth
    iPos=[]
    
    
    
    
    for d in range(depth):
       
        # num of nodes for current depth
        S = N       
        
        
        # 1D example (trans val 1 and 4, merge 2 and 3)
        # 01234567
        # idx: 0, 1, 2, 3
        # val: 1, 2, 3, 4
        
        # merge two leaf nodes or not 
        # mask: False, True, False, False

        # combine two neighbors or transmit
        # combine idx: 1
        # trans idx: 0, 3         
        
        
            
        # merge two leaf nodes or not
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))

        # 2 types of idx for current level of RAHT tree
        # combine two neighbors or transmit
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
       
      
       
        # 2 types of idx for next level of RAHT tree
        # idxT_array, idx of low-freq for next depth level
        # maskT == False for trans (not merge two leaf nodes)
        # maskT == True for comb (merge two leaf nodes)
        # maskT: False, True, False (1D example)
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]
        
        
        # 2 types of weight for next level of RAHT tree
        # wT[N] = wT[M] (not merge two leaf nodes)
        # wT[M] = w[i] + w[j] (merge two leaf nodes)
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        
        
        
        # pos is used to connect C and val/w (current level)
        # posT is used to connect C and val/w (next level)        
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]          
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]   
        
        
        
        
        
       
        
        # transform attr to coeff
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))
        C[pos[left_node_array]], C[pos[right_node_array]] = transform_batched(np.sqrt((w[left_node_array]))/a, 
                                                  np.sqrt((w[right_node_array]))/a, 
                                                  C[pos[left_node_array],None], 
                                                  C[pos[right_node_array],None])
        
        
        
        
        # collect side information for current depth
        parent=np.arange(S)
        parent_t=np.zeros(S)
        parent_t[right_node_array]=1
        parent_t = parent_t.cumsum()
        parent = parent-parent_t    
        # collected but not used in paper 
        iparent.append(parent.astype(int))        
        

        
        
        # High-freq nodes do not exist in the leaf level, thus collect information from the next depth.
        # collect side information after transform for next depth
        iCT_low.append(C[pos[idxT_array]])
        
        num_nodes = N-comb_idx_array.shape[0]
        iW.append(wT[:num_nodes]+0)
        
        Pos_t = val2V(val, d%3)[idxT_array]
        if d%3 == 0:
            Pos_t[:,2]=Pos_t[:,2]//2
        if d%3 == 1:
            Pos_t[:,1]=Pos_t[:,1]//2
        if d%3 == 2:
            Pos_t[:,0]=Pos_t[:,0]//2
        iPos.append(Pos_t) 
        
        

       
        # collect side information of high_freq nodes for next depth
        # tree node feature extraction without considering low-freq nodes
        low_freq[pos[right_node_array]]=C[pos[left_node_array]]    
 
        node_xyz[pos[right_node_array]] = val2V(val[right_node_array], d%3)
        
        if d%3 == 0:
            node_xyz[pos[right_node_array],2]=node_xyz[pos[right_node_array],2]//2
        if d%3 == 1:
            node_xyz[pos[right_node_array],1]=node_xyz[pos[right_node_array],1]//2
        if d%3 == 2:
            node_xyz[pos[right_node_array],0]=node_xyz[pos[right_node_array],0]//2
        
        
        depth_CT[pos[trans_idx_array]] = d
        depth_CT[pos[left_node_array]], depth_CT[pos[right_node_array]] = d, d
        
        # end of information collection
        
                        
        
        
        
        
        # valT, morton code for the next depth
        valT = (val >> 1)[idxT_array]
        
        # num of leaf nodes for next level       
        N_T=N
        N=N-comb_idx_array.shape[0]
        
        
        # move pos,w of high-freq nodes in the end
        # pos: 0, 1, 2, 3
        # posT: 0, 1, 3, *    
        # posT: 0, 1, 3, 2
        
        # transpose
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]        
        
        # move transposed pos,w of high-freq nodes in the end
        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]

        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
        
    outW=np.zeros(w.shape)
    outW[pos]=w
    
    
    low_freq[0] = iCT_low[-1]
    
    
    res = {'CT':C, 
           'w':outW, 
           'depth_CT':depth_CT, 
           'node_xyz':node_xyz,
           'low_freq':low_freq,
           
           'iCT_low':iCT_low,
           'iW':iW,
           'iPos':iPos,
           
           'iparent':iparent,
           }
    
    return res






        
def get_RAHT_tree(inV, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : tree without low- and high-freq coeffs

    '''
    
    
    # N,NN number of points
    # K, dims (3) of geometry    
    N, _ = inV.shape
    NN = N
    

    
    depth *= 3
    
    

    
    wT = np.zeros((N, ))
    valT = np.zeros((N, ))    
    posT = np.zeros((N, )).astype(np.uint64)  
    
        
    
    # morton code and weight for each depth level
    iVAL = np.zeros((depth, N))
    iW = np.zeros((depth, N))
    
    # M, num of nodes for current depth level
    M = N   
    # num of nodes for each depth level
    iM = np.zeros((depth, )).astype(np.uint64)
    
    
    w, val, reord = copyAsort(inV)
    pos = np.arange(N).astype(np.uint64)        
     
    
    
    # construct RAHT tree from bottom to top, similar to RAHT encoding
    # obtain iVAL, iW, iM for RAHT decoding
    for d in range(depth):
        
        iVAL[d,:M] = val[:M]
        iW[d,:M] = w[:M]
        iM[d]= M
        
        M = 0
        S = N
        
        
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False])) 
        
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array]        
        
        wT[np.where(maskT==False)[0]] = w[trans_idx_array]
        posT[np.where(maskT==False)[0]] = pos[trans_idx_array]  
        
        
        wT[np.where(maskT==True)[0]] = w[comb_idx_array]+w[comb_idx_array+1]
        posT[np.where(maskT==True)[0]] = pos[comb_idx_array]  
        
        
        
        
        valT = (val >> 1)[idxT_array]
        
        
        N_T=N
        N=N-comb_idx_array.shape[0]        
        M=N
        
        
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        wT[N_idx_array]=wT[np.where(maskT==True)[0]]                
        posT[N_idx_array]=pos[comb_idx_array+1]
        
 

        pos[N:S] = posT[N:S]
        w[N:S] = wT[N:S]
        
        val, valT = valT, val
        pos, posT = posT, pos
        w, wT = wT, w
        
   
    # input attributes, morton sorted attributes, coeffs
    # inC, C, CT
    # inC and C are connected by reorder
    # C and CT are connected by pos
    
    
    res = {'reord':reord, 
           'pos':pos, 
           'iVAL':iVAL, 
           'iW':iW,
           'iM':iM,
           }
    
    return res    
    
    
    









        
def inv_haar3D(inV, inCT, depth):
    '''
    

    Parameters
    ----------
    inV : point cloud geometry(pre-voxlized and deduplicated)
    inCT : transformed coeffs (high-freq coeffs)
    depth : depth level of geometry(octree)

    Returns
    -------
    res : rec attributes

    '''
    
    
    # N,NN number of points
    # K, dims (3) of geometry    
    N, K = inCT.shape
    NN = N
    

    
    depth *= 3
    
    
    CT = np.zeros((N, K))
    C = np.zeros((N, K))
    outC = np.zeros((N, K))
    
    
    res_tree = get_RAHT_tree(inV, depth)
    reord, pos, iVAL, iW, iM = \
        res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']
    
        
        
    CT = inCT[pos]
    C = np.zeros(CT.shape)
    
 
    
 
    # RAHT decoding from top to bottom
    d = depth
        
    while d:
        
      
        
        d = d-1
        S = iM[d]
        M = iM[d-1] if d else NN 
        
        
        val, w = iVAL[d, :int(S)], iW[d, :int(S)]
            
 
        M = 0
        N = S
        
        
        # get idx, similar to encoding
        temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
        
        mask=temp[:-1]==temp[1:]
        mask=np.concatenate((mask,[False]))
        
        comb_idx_array=np.where(mask==True)[0]
        trans_idx_array=np.where(mask==False)[0]       
        trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
        
        idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
        maskT=mask[idxT_array.astype(int)]
        
        
        # transmit low-freq 
        C[trans_idx_array] = CT[np.where(maskT==False)[0]]
        
        
        # decode low_freq and high_freq to two low_freq coeffs
        
        # N_idx_array, idx of high_freq
        N_T=N
        N=N-comb_idx_array.shape[0] 
        N_idx_array=np.arange(N_T, N, -1)-NN-1
        
        
        left_node_array, right_node_array = comb_idx_array, comb_idx_array+1
        
        a = np.sqrt((w[left_node_array])+(w[right_node_array]))    
        C[left_node_array], C[right_node_array] = itransform_batched(np.sqrt((w[left_node_array]))/a, 
                                        np.sqrt((w[right_node_array]))/a, 
                                        CT[np.where(maskT==True)[0]][:,None], 
                                        CT[N_idx_array.astype(int)][:,None])
        
       
                
    
        CT[:S] = C[:S]
        
  
    outC[reord] = C  
    
    return outC  











if __name__ == "__main__":
    
    depth=9
    
    
    import scipy.io
    import utils
    import time
    
    Qstep = 100
    mat = scipy.io.loadmat('./sample_data/scene0000_00_vh_clean_2.mat')
    
    
    V=mat['V']
    C=mat['C']*255
    
    points, colors = V,C
    
    
    
    
    C_yuv = utils.RGB2YUV(C)

    
    time_raht_encode = time.time()
    res = haar3D(V, C_yuv, depth)
    time_raht_encode_end = time.time()
    print('RAHT encode:', time_raht_encode_end-time_raht_encode)
    
    
    CT_yuv = res['CT']
    CT_yuv_q = np.round(CT_yuv/Qstep)*Qstep
    
    time_raht_decode = time.time()
    C_yuv_rec = inv_haar3D(V, CT_yuv_q, depth)    
    time_raht_decode_end = time.time()
    print('RAHT decode:', time_raht_decode_end-time_raht_decode)
    
    psnr_y = utils.get_PSNR(C_yuv[:,0], C_yuv_rec[:,0])
    print('psnr_y:', psnr_y)
    
    
    
    
    C_rec = utils.YUV2RGB(C_yuv_rec)
    
    
    import pptk
    v=pptk.viewer(V)
    v.attributes(C/255, C_rec/255)
    v.set(point_size=1)    
    
    
    
    
    






