import torch
from torch import nn
from torch.nn import functional as F
from CBAMBlock import CBAM
from BAMBlock import BAM
from SEBlock import SEBlock

# CBAM(channel, reduction)
# BAM(channel, dilation, reduction)
# SEBlock(channel, reduction)
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, use_cbam=False, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ) if in_channel!=out_channel else None

        self.cbam = CBAM(out_channel, 16) if use_cbam else None
        self.se = SEBlock(out_channel, 16) if use_se else None

    def forward(self, input):
        o1 = torch.relu(self.bn1(self.conv1(input)))
        o2 = self.bn2(self.conv2(o1))
        if self.cbam:
            o2 = self.cbam(o2)
        elif self.se:
            o2 = self.se(o2)
        o3 = o2+self.skip(input) if self.skip else o2+input
        return o3




class Resnet18(nn.Module):
    def __init__(self, in_channel, nc, use_bam=False, use_cbam=False, use_se=False):
        super(Resnet18, self).__init__()
        channels = [64,128,256,512]
        self.conv1 = nn.Conv2d(in_channel, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3,2,1)
        in_channel = 64
        self.layer1 = self.make_layers(in_channel, channels[0], 2, use_cbam, use_se)
        self.bam1 = BAM(channels[0], 4, 16) if use_bam else None
        self.layer2 = self.make_layers(channels[0], channels[1], 2, use_cbam, use_se)
        self.bam2 = BAM(channels[1], 4, 16) if use_bam else None
        self.layer3 = self.make_layers(channels[1], channels[2], 2, use_cbam, use_se)
        self.bam3 = BAM(channels[2], 4, 16) if use_bam else None
        self.layer4 = self.make_layers(channels[2], channels[3], 2, use_cbam, use_se)
        self.classifier = nn.Linear(channels[3], nc)

    def forward(self, input):
        o1 = torch.relu(self.bn1(self.conv1(input)))
        o1 = self.maxpool(o1)
        o1 = self.layer1(o1)
        if self.bam1:
            o1 = self.bam1(o1)
        o2 = self.layer2(o1)
        if self.bam2:
            o2 = self.bam2(o2)
        o3 = self.layer3(o2)
        if self.bam3:
            o3 = self.bam3(o3)
        o4 = self.layer4(o3)
        o4 = F.adaptive_avg_pool2d(o4, (1,1))
        o4 = o4.view(o4.shape[0], -1)
        logits = self.classifier(o4)
        return logits

    def make_layers(self, in_channel, out_channel, depth, use_cbam=False, use_se=False):
        blocks = []
        for i in range(depth):
            if not i:
                blocks.append(BasicBlock(in_channel, out_channel, 2 if in_channel!=out_channel else 1, use_cbam, use_se))
            else:
                blocks.append(BasicBlock(out_channel, out_channel, 1, use_cbam, use_se))
        return nn.Sequential(*blocks)

# net = Resnet18(3,2,use_cbam=True)
# net.eval()
# input = torch.rand(1,3,256,256)
# out = net(input)
# print(out.shape)