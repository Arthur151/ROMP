import torch 
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(S, input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        #Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )

class TemporalPoseRegressor(nn.Module):
    def __init__(self, S = 8,
                 in_dim=128, n_jts_out=22, init_pose=None, jt_dim=6,
                 dim=512, depth=2, heads=8, dim_head=64, mlp_dim=512, dropout=0.1,
                 share_regressor=1,
                 *args, **kwargs):
        super(TemporalPoseRegressor, self).__init__()
        
        output_dim = n_jts_out * jt_dim
        self.regressor = MLPMixer(S, in_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.)

    def forward(self, x, mask=None):
        """
        Args:
            - x: torch.Tensor - torch.float32 - [batch_size, seq_len, 128]
            - mask: torch.Tensor - torch.bool - [batch_size, seq_len]
        Return:
            - y: torch.Tensor - [batch_size, seq_len, 24*6] - torch.float32
        """
        y=self.regressor(x)

        return y
    

class TemporalSMPLShapeRegressor(nn.Module):
    def __init__(self, S=8,
                 in_dim=128, out_dim=21, init_shape=None,
                 dim=256, depth=1, heads=8, dim_head=64, mlp_dim=512, dropout=0.1,
                 share_regressor=1,
                 *args, **kwargs):
        super(TemporalSMPLShapeRegressor, self).__init__()
        self.regressor = MLPMixer(S, in_dim, dim, out_dim, depth, expansion_factor=4, dropout=0.)

    def forward(self, x, mask=None):
        """
        Args:
            - x: torch.Tensor - torch.float32 - [batch_size, seq_len, 128]
            - mask: torch.Tensor - torch.bool - [batch_size, seq_len]
        Return:
            - y: torch.Tensor - [batch_size, seq_len, 24*6] - torch.float32
        """
        y=self.regressor(x)

        return y

if __name__ == '__main__':
    model = MLPMixer(
            S=8,
            input_dim=128,
            dim=512,
            output_dim=166,
            depth=6).cuda()
    x = torch.rand(8, 8, 128).cuda()
    y = model(x)
    print(y.shape)
    