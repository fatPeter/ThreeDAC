import torch
import torch.nn as nn
from models.spvcnn_if import SPVCNN_if




class InitialCodingModule(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super().__init__()
        self.High_Freq_Node_Extractor = nn.ModuleList([nn.Sequential(nn.Linear(in_channel, out_channel)) for _ in range(depth*3)]) 
        
        self.Low_Freq_Node_Spatial_Extractor = SPVCNN_if(in_channel=in_channel, out_channel=out_channel, depth=depth)
        self.Low_Freq_Node_Extractor = nn.ModuleList([nn.Sequential(nn.Linear(out_channel, out_channel)) for _ in range(depth*3)]) 
        
        
        
    def spatial_aggregate(self, x, query_points):
        
        low_freq_info = self.Low_Freq_Node_Spatial_Extractor(x, query_points)
        return low_freq_info
    
    
    def forward(self, high_freq_info, low_freq_info, depth_idx):
        
        context_high = self.High_Freq_Node_Extractor[depth_idx](high_freq_info)
        context_low = self.Low_Freq_Node_Extractor[depth_idx](low_freq_info)
        
        context_initial_coding = torch.cat((context_high, context_low), -1)
        return context_initial_coding
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    







