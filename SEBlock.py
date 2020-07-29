import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = nn.functional.adaptive_avg_pool2d(input, (1,1))
        x = x.view(x.shape[0], x.shape[1])
        x = self.excitation(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return input * x
