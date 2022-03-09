import torch
import torch.nn as nn

from multiview_detector.models.attn_module import ExpandedChannelGate, ChannelGate


class CutoffModule(nn.Module):
    def __init__(self, input_dim, depth_scales):
        super().__init__()
        self.input_dim = input_dim
        self.depth_scales = depth_scales

        self.channel_attn = ChannelGate(self.input_dim) if self.depth_scales == 1 else ExpandedChannelGate(self.input_dim, self.depth_scales)
    
    
    def forward(self, x):
        N, C, _, _ = x.shape
        block_size = C // self.depth_scales

        if self.depth_scales == 1:
            out_feat = self.channel_attn(x)
        else:
            attn = self.channel_attn(x)
            values, indices = torch.topk(attn, block_size, dim=1)
            indices = indices.squeeze(-2).squeeze(-2)
            out_feat = torch.cat([torch.cat([torch.index_select(x[j], 0, indices[:, :, i][j]).unsqueeze(0) for j in range(N)], dim=0) for i in range(self.depth_scales)], dim=1)

        return out_feat