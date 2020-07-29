import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction):
        super(ChannelAttention, self).__init__()
        self.l1 = nn.Linear(in_channel, in_channel//reduction)
        self.l2 = nn.Linear(in_channel//reduction, in_channel)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, input):
        input = F.adaptive_avg_pool2d(input, (1,1))
        input = input.view(input.shape[0], -1)
        out = self.l2(self.l1(input))
        out = self.bn(out.view(*out.shape,1,1))
        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channel, dilation, reduction):
        super(SpatialAttention, self).__init__()
        channel = in_channel // reduction
        self.conv1 = nn.Conv2d(in_channel, channel, 1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=dilation, bias=False, dilation=dilation)
        self.conv3 = nn.Conv2d(channel, channel, 3, padding=dilation, bias=False, dilation=dilation)
        self.conv4 = nn.Conv2d(channel, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, input):
        o1 = self.conv1(input)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        out = self.bn(o4)
        return out

class BAM(nn.Module):
    def __init__(self, in_channel, dilation, reduction):
        super(BAM, self).__init__()
        self.channelAttention = ChannelAttention(in_channel, reduction)
        self.spatialAttention = SpatialAttention(in_channel, dilation, reduction)

    def forward(self, input):
        ca = self.channelAttention(input)
        sa = self.spatialAttention(input)
        out = torch.sigmoid(ca+sa)
        return (out+1)*input
