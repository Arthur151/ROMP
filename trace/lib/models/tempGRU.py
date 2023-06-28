import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            with_gru=False,
            input_size=128,
            out_size=[6],
            n_gru_layers=1,
            hidden_size=128):
        super(TemporalEncoder, self).__init__()
        # self.with_gru=with_gru
        # if self.with_gru:
        #     self.gru = nn.GRU(
        #         input_size=input_size,
        #         hidden_size=input_size,
        #         bidirectional=False,
        #         num_layers=n_gru_layers,
        #         batch_first=True)
            
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=True))
        
        self.out_layers = nn.ModuleList([nn.Linear(hidden_size, size) for size in out_size])
        
    def forward(self, x, *args):
        n,t,f = x.shape
        # if self.with_gru:
        #     y, h = self.gru(x)
        #     x = y + x

        y = self.regressor(x.reshape(-1,f))
        y = torch.cat([self.out_layers[ind](y) for ind in range(len(self.out_layers))], -1)
        return y.reshape(n,t,-1)
    
    def image_forward(self, x):
        # x in shape batch, feature_dim
        y = self.regressor(x)
        y = torch.cat([self.out_layers[ind](y) for ind in range(len(self.out_layers))], -1)
        return y