import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
"""
Brought from https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
"""

class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=2):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i]))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, batch_first=True, return_all_layers=False):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        drop_first = False
        # Implement stateful ConvLSTM
        if hidden_state is None:
            #hidden_state = self._init_hidden_first_frame(input_tensor)
            hidden_state = self._init_zero_hidden(input_tensor=input_tensor)
            input_tensor = torch.cat([input_tensor[:,[0]], input_tensor], 1)
            drop_first = True

        layer_output_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            output_inner = [None for _ in range(seq_len)]
            h = hidden_state[layer_idx]
            time_ids = torch.arange(seq_len)
            for t in time_ids:
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](h, cur_layer_input[:, t, :, :, :])
                output_inner[t] = h
            hidden_state[layer_idx] = h
            
            # bi-directional can't predict in clip mode, cause the hidden state is not right between frames.
            # output_inner = [None for _ in range(seq_len)]
            # if layer_idx%2==0:
            #     h = hidden_state[layer_idx//2]
            #     time_ids = torch.arange(seq_len)
            # else:
            #     h = layer_output_list[-1][:,seq_len-1]
            #     time_ids = seq_len-torch.arange(seq_len)-1
            # for t in time_ids:
            #     # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
            #     h = self.cell_list[layer_idx](h, cur_layer_input[:, t, :, :, :])
            #     output_inner[t] = h
            # if layer_idx%2==0:
            #     hidden_state[layer_idx//2] = h
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)

        if not return_all_layers:
            layer_output_list = layer_output_list[-1]
        if drop_first:
            layer_output_list = layer_output_list[:,1:]

        return layer_output_list, hidden_state

    def _init_hidden_first_frame(self, input_tensor):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(input_tensor[:,0].clone().detach())
        return init_states
    
    def _init_zero_hidden(self, input_tensor):
        init_states = []
        for i in range(self.num_layers):
            zero_hidden = torch.zeros_like(input_tensor[:,0])
            if zero_hidden.shape[1] == self.hidden_dim:
                zero_hidden = zero_hidden[:,[0]].repeat(1, self.hidden_dim, 1, 1, 1)
            init_states.append(zero_hidden)
        return init_states
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        # When both f1 and f2 are applied SS-Trans, corr_multiplier = 2.
        # Otherwise corr_multiplier = 1.
        cor_planes = args.corr_levels * \
            args.corr_multiplier * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(input_dim=hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net_feat, inp_feat, corr, flow, upsample=True):
        # motion_features: (256+2)-dimensional.
        motion_features = self.encoder(flow, corr)
        inp_feat = torch.cat([inp_feat, motion_features], dim=1)

        net_feat = self.gru(net_feat, inp_feat)
        delta_flow = self.flow_head(net_feat)

        # scale mask to balance gradients
        mask = .25 * self.mask(net_feat)
        return net_feat, mask, delta_flow


#from lib.models.eca_setrans import ExpandedFeatTrans
#from torch import nn, einsum
#from einops import rearrange
# Aggregate output is dim-dimensional, same as the input. No FFN is used.


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads=4,
        dim_head=128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        # project is None for GMA.
        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out

class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim,
                              input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.use_setrans = args.use_setrans
        if self.use_setrans:
            self.intra_trans_config = args.intra_trans_config
            self.aggregator = ExpandedFeatTrans(
                self.intra_trans_config, 'Motion Aggregator')
        else:
            # Aggregate is attention with a (learnable-weighted) skip connection, without FFN.
            self.aggregator = Aggregate(
                args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)

    def forward(self, net, inp, corr, flow, attention):
        # encoder: BasicMotionEncoder
        # corr: [3, 676, 50, 90]
        motion_features = self.encoder(flow, corr)
        # motion_features: 128-dim
        if self.use_setrans:
            # attention is multi-mode. ExpandedFeatTrans takes multi-mode attention.
            B, C, H, W = motion_features.shape
            motion_features_3d = motion_features.view(
                B, C, H*W).permute(0, 2, 1)
            # motion_features_3d: [1, 7040, 128], attention: [1, 4, 7040, 7040]
            motion_features_global_3d = self.aggregator(
                motion_features_3d, attention)
            motion_features_global = motion_features_global_3d.view(
                B, H, W, C).permute(0, 3, 1, 2)
        else:
            # attention: [8, 1, 2852, 2852]. motion_features: [8, 128, 46, 62].
            motion_features_global = self.aggregator(
                attention, motion_features)

        inp_cat = torch.cat(
            [inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


if __name__ == '__main__':
    inputs = torch.ones(2,3,2,8,8).cuda().float()
    gru = ConvGRU(input_dim=2, hidden_dim=2, kernel_size=3, num_layers=2).cuda()
    outs = gru(inputs,return_all_layers=False)