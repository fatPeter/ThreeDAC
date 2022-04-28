import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

os.chdir("..") 


import numpy as np

from Haar3D_inform import haar3D, inv_haar3D, copyAsort, get_RAHT_tree, itransform_batched
import torch
import torch.optim as optim


from dataloaders.semantic_kitti_pcc import SemanticKittiPCC
import time

import utils
from InitialCoding import InitialCoding_SemanticKitti as InitialCoding
from InitialCoding import Construct_InitialCoding_Info_SemanticKitti_depth as Construct_InitialCoding_Info_depth



parser = argparse.ArgumentParser()
parser.add_argument('--Qstep', type=int, default=10)
parser.add_argument('--dir_path', type=str, default='dataset/SemanticKitti/')
parser.add_argument('--depth', type=int, default=12)
parser.add_argument('--step', type=int, default=20)
opt = parser.parse_args()
print(opt)






dir_path=opt.dir_path
out_model_dir=os.path.join(r'checkpoints/', BASE_DIR.split('/')[-1])

if not os.path.exists(out_model_dir):
    os.makedirs(out_model_dir)



depth = opt.depth
Qstep = opt.Qstep
step = opt.step


    








train_dataset = SemanticKittiPCC(dir_path=dir_path, mode='train', depth=depth)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0) 

test_dataset = SemanticKittiPCC(dir_path=dir_path, mode='test', depth=depth)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)        



train_dataset.data_path_list=train_dataset.data_path_list[::step]
test_dataset.data_path_list=test_dataset.data_path_list[::step]



from models.EntropyBottleneck import EntropyBottleneck

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entropy_bottleneck = EntropyBottleneck(channels=1, content_chann=16).cuda()





from models.InitialCodingModule import InitialCodingModule

initial_coding_module = InitialCodingModule(in_channel=4, out_channel=8, depth=depth).cuda()



model_path=out_model_dir+r'/eb_q%d_best.pth'%(Qstep)
initial_coding_module_path=out_model_dir+r'/initial_coding_module_q%d_best.pth'%(Qstep) 


entropy_bottleneck.load_state_dict(torch.load(model_path))
initial_coding_module.load_state_dict(torch.load(initial_coding_module_path))










with torch.no_grad():
    test_loss_sum = 0
    
    bpv_list = []
    bpp_list = []
    time_list = []     
    psnr_list = []
    
    entropy_bottleneck.eval() 
    initial_coding_module.eval() 
    
    
 
    
    training=False       
    
    for batch_id, data in enumerate(test_dataloader):
        print(batch_id)
                   
        bpv_sum = 0
        bpp_sum = 0                
        
            
        
        # load point cloud
        points, colors = data
        points, colors = points[0].numpy(), colors[0].numpy()
        colors = colors*255
        
        
        colors_ori = colors
        colors = np.zeros(colors.shape)
        
        res_temp = InitialCoding(points, colors_ori, depth, Qstep)
        DC_coef_ori = torch.round(res_temp['CT'][0]/Qstep)*Qstep
        AC_coef_ori = torch.round(res_temp['CT'][1:]/Qstep)*Qstep
        
        
        # inCT = np.zeros(colors.shape)
        # inCT[0] = DC_coef_ori           
        
        
        res = InitialCoding(points, colors, depth, Qstep)

        # merge this part to get_RAHT_tree
        w_ad, depth_CT_ad, node_xyz_ad = res['w'], res['depth_CT'], res['node_xyz']
        iW_ad, iPos_ad = res['iW'], res['iPos']
                
        
        res_tree = get_RAHT_tree(points, depth)
        reord, pos, iVAL, iW, iM = \
            res_tree['reord'], res_tree['pos'], res_tree['iVAL'], res_tree['iW'], res_tree['iM']       
            
        # RAHT inverse transform
        d = depth*3   
        N, K = colors.shape
        NN = N
        C = np.zeros((N, K))
        CT = np.zeros((N, K))
        outC = np.zeros((N, K))     
        
        
  
        
        num_head = 0
        num_trans = 1
        for i in range(d):
            mask = res['depth_CT']==i
            mask = mask[1:]
            if torch.sum(mask)==1:
                num_trans+=1
            if torch.sum(mask)>1:
                num_head+=1
                
        head_tag = num_head-1
        trans_tag = num_trans-1
        
        
        
        base=os.path.basename(test_dataloader.dataset.data_path_list[batch_id])
        file_name = os.path.splitext(base)[0]
        bin_dirpath = os.path.join(BASE_DIR.split('/')[-1], 'bin', file_name)        
                
        head_binname = bin_dirpath+'/head.bin'
        with open(head_binname, 'rb') as f:
            head_array = np.frombuffer(f.read(num_head*2*4), dtype=np.int16) 
        head_array = head_array.reshape((-1,2))
            
            
        trans_binname = bin_dirpath+'/trans.bin'
        with open(trans_binname, 'rb') as f:
            trans_array = np.frombuffer(f.read(num_trans*1*4), dtype=np.int16)     
        trans_array = trans_array.reshape((-1,1))
        
        CT[0] = trans_array[0].astype(float)*Qstep
        

                
        

        while d:
            
          
            
            d = d-1
            S = iM[d]
            M = iM[d-1] if d else NN 
            
                  
            
            
            #print(M)
            
            val, w = iVAL[d, :int(S)], iW[d, :int(S)]
                
     
            M = 0
            N = S
            i = 0     
            
            temp=val.astype(np.uint64)&0xFFFFFFFFFFFFFFFE
            
            mask=temp[:-1]==temp[1:]
            mask=np.concatenate((mask,[False]))
            
            comb_idx_array=np.where(mask==True)[0]
            trans_idx_array=np.where(mask==False)[0]       
            trans_idx_array=np.setdiff1d(trans_idx_array, comb_idx_array+1)
            
            idxT_array=np.setdiff1d(np.arange(S), comb_idx_array+1)
            maskT=mask[idxT_array.astype(int)]
            
            
            
            C[trans_idx_array] = CT[np.where(maskT==False)[0]]
            
            
            
            N_T=N
            N=N-comb_idx_array.shape[0] 
            N_idx_array=np.arange(N_T, N, -1)-NN-1
            
       
            
            mask_ad = depth_CT_ad==d
            mask_ad = mask_ad[1:]
            if torch.sum(mask_ad)>1:
              
                
                # update side information from current depth level
                res['low_freq'][1:][mask_ad] = torch.tensor(
                    CT[np.where(maskT==True)[0]])
                res['iCT_low'][d] = torch.tensor(CT[:iM[d+1]]+0)
                
                
                
                def low_freq2attri(low_freq, w):
                    yuv = low_freq/np.sqrt(w[:,None])
                    return torch.tensor(yuv)
                
                
                low_freq = res['low_freq'][1:][mask_ad]       
                low_freq[low_freq<=0]=0.1   
                res['low_freq'][1:][mask_ad] = low_freq
                res['cyuv'][1:][mask_ad] = low_freq2attri(low_freq, w_ad[1:][mask_ad])
                
                
                
                iCT_low_d = res['iCT_low'][d]
                iCT_low_d[iCT_low_d<=0]=0.1   
                res['iCT_low'][d] = iCT_low_d
                res['iyuv'][d] = low_freq2attri(iCT_low_d, iW_ad[d])
                
                
                
                
                    
                # convert side information to context tensor
                res_ini_info = Construct_InitialCoding_Info_depth(res, Qstep, depth_idx=d)
                high_freq_info = res_ini_info['high_freq_info']
                high_freq_nodes_sp = res_ini_info['high_freq_nodes_sp']
                low_freq_nodes_sp = res_ini_info['low_freq_nodes_sp']    
                
                
                # res_inter = Construct_InterChannel_Info_depth(res, Qstep, depth_idx=i)
                # inter_channel_info_y = res_inter['inter_channel_info_y']
                # inter_channel_info_yu = res_inter['inter_channel_info_yu']
                
                # high_freq_nodes_sp = res_inter['high_freq_nodes_sp']    
                # high_freq_nodes_y_sp = res_inter['high_freq_nodes_y_sp']    
                # high_freq_nodes_yu_sp = res_inter['high_freq_nodes_yu_sp'] 
                
                
                
                    
                                 
                low_freq_info = initial_coding_module.spatial_aggregate(low_freq_nodes_sp.cuda(), 
                                                                        high_freq_nodes_sp.cuda()
                                                                        )
                
                # spatial_info_y, spatial_info_yu = inter_channel_module.spatial_aggregate(high_freq_nodes_y_sp.cuda(),
                #                                                                          high_freq_nodes_yu_sp.cuda(),
                #                                                                          high_freq_nodes_sp.cuda()
                #                                                                          )
            
                
                context_initial_coding = initial_coding_module(high_freq_info.cuda(), 
                                                             low_freq_info, 
                                                             depth_idx=d)
                # context_inter_channel = inter_channel_module(inter_channel_info_y.cuda(), 
                #                                              inter_channel_info_yu.cuda(), 
                #                                              spatial_info_y, 
                #                                              spatial_info_yu, 
                #                                              depth_idx=i)
                
             
               
                    
                min_v, max_v = head_array[head_tag]
                head_tag = head_tag-1
                
                if min_v == max_v:
                    CT_q_dec = min_v
                else:
                    # load bitstream
                    strings_list=[]
                    for yuv_idx in range(colors.shape[-1]):
                        wavelet_binname = bin_dirpath+'/wavelet_%02d%d.bin'%(d,yuv_idx)
                        with open(wavelet_binname, 'rb') as f:
                            strings = f.read()    
                        strings_list.append(strings)                    
                            
                        shape = colors[1:][mask_ad].shape  
    
                    
                    # res_temp['CT'][res_temp['depth_CT']==d] for validate
                    
                    # first channel
                    channel_idx = 0
                    
                    
                    context = context_initial_coding[:,None]     
                    CT_q_dec = entropy_bottleneck.decompress(strings_list[channel_idx], min_v, max_v, shape, context, channel_idx, device=device)
                    
                 
                
                
                
                CT[N_idx_array.astype(int)] = CT_q_dec*Qstep
                
                # temp = np.concatenate((input_yu, CT_q_dec),-1)*Qstep-AC_coef_ori[mask_ad].numpy()
                # if np.sum(temp**2)!=0:
                #     print(d, 'error')
                
                
                
                     
            elif torch.sum(mask_ad)==1:     
                trans_coeffs = trans_array[trans_tag]
                trans_tag = trans_tag-1
                CT[N_idx_array.astype(int)] = trans_coeffs*Qstep
                
                # CT[N_idx_array.astype(int)] = AC_coef_ori[mask_ad]   
            else:
                pass
            
            
            
            
            
            
            i=comb_idx_array
            j=i+1  
            
            a = np.sqrt((w[i])+(w[j]))    
            C[i], C[j] = itransform_batched(np.sqrt((w[i]))/a, 
                                            np.sqrt((w[j]))/a, 
                                            CT[np.where(maskT==True)[0]][:,None], 
                                            CT[N_idx_array.astype(int)][:,None])
            
            CT[:S] = C[:S]
            
            
            # res['low_freq'][1:][mask_ad] = CT[np.where(maskT==True)[0]]
            
            
            
            
            
      
        outC[reord] = C  
        
        
        psnr = utils.get_PSNR(colors_ori[:,0], outC[:,0])
        print('psnr:', psnr)
        
        
        psnr_list.append(psnr)
        
    
        # break
    print('psnr', np.mean(psnr_list))
        
        

  
# import pptk
# v=pptk.viewer(points)
# v.attributes(colors_ori[:,0]/255, outC[:,0]/255)
# v.set(point_size=1)    

        











