import torch
from torch import nn
import torch.nn.functional as F

'''通道注意力模块'''
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            # nn.ReLU(),
            nn.Linear(channel//reduction, channel, bias=False)
        )

    def forward(self, input):
        avg_out = self.mlp(F.adaptive_avg_pool2d(input, (1,1)).view(input.shape[0], input.shape[1]))
        max_out = self.mlp(F.adaptive_max_pool2d(input, (1, 1)).view(input.shape[0], input.shape[1]))
        out = torch.sigmoid(avg_out + max_out)
        out = out.unsqueeze(-1).unsqueeze(-1)
        return out

'''空间注意力模块'''
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, stride=1, padding=3, bias=False)

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)

'''注意力模块CBAM'''
class CBAM(nn.Module):
    def __init__(self, channel, reduction):
        super(CBAM, self).__init__()
        self.channelAttention = ChannelAttention(channel, reduction)
        self.saptialAttention = SpatialAttention()

    def forward(self, input):
        # x = input
        out = self.channelAttention(input)
        input = out * input
        out = self.saptialAttention(input)
        out = out * input
        return out
