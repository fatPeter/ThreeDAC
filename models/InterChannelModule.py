import torch
import torch.nn as nn
from models.spvcnn_if import SPVCNN_if
import torch.nn.functional as F


            


class InterChannelModule(nn.Module):
    def __init__(self, out_channel, depth):
        super().__init__()
        self.Coef_Extractor_y = nn.ModuleList([nn.Sequential(nn.Linear(1, out_channel)) for _ in range(depth*3)])
        self.Coef_Extractor_yu = nn.ModuleList([nn.Sequential(nn.Linear(2, out_channel)) for _ in range(depth*3)])
        
        self.Spatial_Agg_y = SPVCNN_if(in_channel=1, out_channel=out_channel, depth=depth, cr=0.05).cuda()
        self.Spatial_Agg_yu = SPVCNN_if(in_channel=2, out_channel=out_channel, depth=depth, cr=0.05).cuda()
        
        self.Spatial_Extractor_y = nn.ModuleList([nn.Sequential(nn.Linear(out_channel, out_channel)) for _ in range(depth*3)])
        self.Spatial_Extractor_yu = nn.ModuleList([nn.Sequential(nn.Linear(out_channel, out_channel)) for _ in range(depth*3)])
  
        
  
    def spatial_aggregate(self, x_y, x_yu, query_points):
        
        spatial_info_y = self.Spatial_Agg_y(x_y, query_points)
        spatial_info_yu = self.Spatial_Agg_yu(x_yu, query_points)
        
        return spatial_info_y, spatial_info_yu
        
        
        
        
    def forward(self, coef_y, coef_yu, spatial_y, spatial_yu, depth_idx):
        # x: SparseTensor query_points: SparseTensor z: PointTensor
        
        context_coef_y = self.Coef_Extractor_y[depth_idx](coef_y)
        context_spatial_y = self.Spatial_Extractor_y[depth_idx](spatial_y)
        context_y = torch.cat((context_coef_y, context_spatial_y) ,-1)[:,None]
        
        context_coef_yu = self.Coef_Extractor_yu[depth_idx](coef_yu)
        context_spatial_yu = self.Spatial_Extractor_yu[depth_idx](spatial_yu)
        context_yu = torch.cat((context_coef_yu, context_spatial_yu) ,-1)[:,None] 
        
        
        context_inter_channel = torch.cat((torch.zeros(context_y.shape).cuda(), 
                                context_y, 
                                context_yu), 1)    
        
    
        return context_inter_channel
    
    
    def extract_channel(self, coef_y, x_y, query_points, depth_idx, channel_idx):
        # x: SparseTensor query_points: SparseTensor z: PointTensor
        

        if channel_idx == 1:
            context_coef_y = self.Coef_Extractor_y[depth_idx](coef_y)
            spatial_info_y = self.Spatial_Agg_y(x_y, query_points)
            context_spatial_y = self.Spatial_Extractor_y[depth_idx](spatial_info_y)
            context_y = torch.cat((context_coef_y, context_spatial_y) ,-1)[:,None]
            return context_y
        elif channel_idx == 2:
            context_coef_yu = self.Coef_Extractor_yu[depth_idx](coef_y)
            spatial_info_yu = self.Spatial_Agg_yu(x_y, query_points)
            context_spatial_yu = self.Spatial_Extractor_yu[depth_idx](spatial_info_yu)
            context_yu = torch.cat((context_coef_yu, context_spatial_yu) ,-1)[:,None] 
            return context_yu    
        else:
            print('incorrect channel idx')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







