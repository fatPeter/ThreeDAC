import torch
import torch.utils.data as data
import numpy as np
import os
import glob
import trimesh
import open3d as o3d



def rotate_alongz(data,angle):
    rotation_angle = angle
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    return np.matmul(data,rotation_matrix)



class SemanticKittiPCC(data.Dataset):
    def __init__(self, dir_path, mode, depth, aug=False):
        self.dir_path = os.path.join(dir_path, 'sequences')
        self.depth = depth
        
        if mode=='train':
            self.seqs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']                        
            self.data_path_list = []
            for seq in self.seqs:
                seq_files = sorted(
                    os.listdir(os.path.join(self.dir_path, seq, 'velodyne')))
                seq_files = [
                    os.path.join(self.dir_path, seq, 'velodyne', x) for x in seq_files
                ]
                self.data_path_list.extend(seq_files)                      
        if mode == 'test':
            self.seqs = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20','21']    
            self.data_path_list = []
            for seq in self.seqs:
                seq_files = sorted(
                    os.listdir(os.path.join(self.dir_path, seq, 'velodyne')))
                seq_files = [
                    os.path.join(self.dir_path, seq, 'velodyne', x) for x in seq_files
                ]
                self.data_path_list.extend(seq_files)               
            
        self.aug = aug
     
 
        
    def __getitem__(self, index):
        
        with open(self.data_path_list[index], 'rb') as b:
            pc = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        
        
        points=pc[:,:3]
        colors=pc[:,3:4]
        
        if self.aug:
            angle = np.random.uniform(0, 2*np.pi)
            points = rotate_alongz(points,angle)
            
        
        
        
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
    
    
    dir_path=r'../dataset/SemanticKitti/'
    
    train_dataset = SemanticKittiPCC(dir_path=dir_path, mode='test', depth=12, aug=False)
    
    
    points, colors = train_dataset[0]
    
    
    import pptk
    v = pptk.viewer(points,debug=True)
    v.attributes(colors[:,0])
    v.set(point_size=0.5) 
    
    
    
    
    
    
    
    
    
    
    
        