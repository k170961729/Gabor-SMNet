import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from gcn.modules import GConv


def convbn_gc(in_planes, out_planes, kernel_size, nChannel, nScale, stride, pad, dilation, expand=False):

    return nn.Sequential(GConv(in_planes, out_planes, kernel_size=kernel_size, M=nChannel, nScale=nScale, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False, expand=expand),
                         nn.BatchNorm2d(out_planes*nChannel))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):


    def __init__(self, inplanes, planes, nChannel, nScale, stride, pad, downsample, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn_gc(inplanes, planes,3, nChannel, nScale, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_gc(planes, planes, 3, nChannel, nScale, 1, pad, dilation)

        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    # (D, softmax(d), H, W)
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1 , x.size()[2], x.size()[3])
        out = torch.sum(x*disp, 1)
        return out 


class feature_extraction(nn.Module):
    def __init__(self, channel=4):
        super(feature_extraction, self).__init__()
        self.inplanes = 8
        self.channel = channel
        self.firstconv = nn.Sequential(convbn_gc(3,8,5, channel, 4,2,2,1,True),
                                       nn.ReLU(inplace = True))

        self.layer1 = self._make_layer(BasicBlock, 8, 3, 1, channel, 4, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 16, 8, 2, channel, 4, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 32, 3, 1, channel, 4, 1, 2)



        self.lastconv = nn.Conv2d(128, 32, kernel_size=3, padding=1, stride = 1, bias=False)


    def _make_layer(self, block, planes, blocks, stride, nChannel, nScale, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(GConv(self.inplanes, planes, kernel_size=1, stride=stride, bias=False, M=nChannel, nScale=nScale),
                                       nn.BatchNorm2d(planes*nChannel),)
        layers = []
        layers.append(block(self.inplanes, planes, nChannel, nScale, stride, pad, downsample, dilation))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nChannel, nScale, 1, pad, None, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output_feature = self.lastconv(output)

        return output_feature



