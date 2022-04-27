import torch
import torch.utils.data as data
import numpy as np
import os
import glob
import open3d as o3d


def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines


class ScannetPCC(data.Dataset):
    def __init__(self, dir_path, mode, depth):
        self.dir_path = dir_path
        self.depth = depth
        
        if mode=='train':
            data_paths = read_txt(os.path.join(dir_path, r'scannet_v2_train.txt'))
            self.data_path_list = [os.path.join(dir_path, 'scans', data_path, data_path+'_vh_clean_2.ply')  
                                   for data_path in data_paths]     
            self.data_path_list = sorted(self.data_path_list)
            
        if mode=='val':
            data_paths = read_txt(os.path.join(dir_path, r'scannet_v2_val.txt'))
            self.data_path_list = [os.path.join(dir_path, 'scans', data_path, data_path+'_vh_clean_2.ply')  
                                   for data_path in data_paths]                       
            self.data_path_list = sorted(self.data_path_list)
 
        
    def __getitem__(self, index):
        path=self.data_path_list[index]
        pc=o3d.io.read_point_cloud(path)
        
        points=np.asarray(pc.points)
        colors=np.asarray(pc.colors) 
        
        
        
        points=points-points.min(0)
        points=points/points.max()
        
        center=(points.max(0)+points.min(0))/2
        points=points-center+np.array([0.5,0.5,0.5])
        
        
        points=points*(2**self.depth-1)
        points=np.round(points)
        
        
        points, idx=np.unique(points, return_index=True, axis=0)
        colors=colors[idx]   
       
        return points, colors
           

    def __len__(self):
        return len(self.data_path_list)



    
    
if __name__ == '__main__':
    
    
    dir_path=r'../dataset/Scannet_v2/'
    
    train_dataset = ScannetPCC(dir_path=dir_path, mode='train', depth=9)
    points, colors = train_dataset[0]
    
    
    import pptk
    v = pptk.viewer(points,debug=True)
    v.attributes(colors)
    v.set(point_size=0.5) 
    
    
    
    
    
    
    
    
    
    
    
        