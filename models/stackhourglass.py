import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule import *


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()

        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.d0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),   
                                nn.ReLU(inplace=True),
                                convbn_3d(32, 32, 3, 1, 1))


        self.d1 = nn.Sequential(convbn_3d(32, 64, 3, 2, 1),    
                                nn.ReLU(inplace=True),
                                convbn_3d(64, 64, 3, 1, 1))


        self.d2 = nn.Sequential(convbn_3d(64, 64, 3, 2, 1),    
                                    nn.ReLU(inplace=True),
                                    convbn_3d(64, 64, 3, 1, 1))


        self.d3 = nn.Sequential(convbn_3d(64, 128, 3, 2, 1),   
                                    nn.ReLU(inplace=True),
                                    convbn_3d(128, 128, 3, 1, 1))


        self.trans_conv1 = nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(64))

        self.trans_conv2 = nn.Sequential(nn.ConvTranspose3d(64, 64, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(64))

        self.trans_conv3 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(32))

        self.trans_conv4 = nn.Sequential(nn.ConvTranspose3d(32, 32, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                         nn.BatchNorm3d(32))

        self.trans_conv5 = nn.Sequential(nn.ConvTranspose3d(32, 1, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif  isinstance(m, GConv):
                init.xavier_uniform_(m.weight, gain = np.sqrt(2))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        ref_img_fea = self.feature_extraction(left)
        target_img_fea = self.feature_extraction(right)


        #cost volume
        cost = Variable(torch.FloatTensor(ref_img_fea.size()[0], ref_img_fea.size()[1]*2, self.maxdisp//4,  ref_img_fea.size()[2],  ref_img_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp//4):
            if i > 0:
                cost[:, :ref_img_fea.size()[1], i, :,i:] = ref_img_fea[:,:,:,i:]
                cost[:, ref_img_fea.size()[1]:, i, :,i:] = target_img_fea[:,:,:,:-i]
            else:
                cost[:, :ref_img_fea.size()[1], i, :,:] = ref_img_fea
                cost[:, ref_img_fea.size()[1]:, i, :,:] = target_img_fea

        #returns a contiguous tensor containing the same data as self tensor.
        #If self tensor is contiguous, this function returns the self tensor.

        cost = cost.contiguous()

        cost0 = self.d0(cost)
        cost0_1 = F.relu(cost0, inplace=True) 

        cost1 = self.d1(cost0_1)  
        cost1_1 = F.relu(cost1, inplace=True)


        cost2 = self.d2(cost1_1)  
        cost2_1 = F.relu(cost2, inplace=True)

        cost3 = self.d3(cost2_1)   
        cost3_1 = F.relu(cost3, inplace=True)


        cost4 = self.trans_conv1(cost3_1)  

        cost4_relu = F.relu(cost4 + cost2, inplace=True)

        cost5 = self.trans_conv2(cost4_relu)  
        cost5_relu = F.relu(cost5 + cost1, inplace=True)

        cost6 = self.trans_conv3(cost5_relu) 
        cost6_relu = F.relu(cost6 + cost0, inplace=True)


        cost7 = self.trans_conv4(cost6_relu)  
        cost7_relu = F.relu(cost7, inplace=True)

        cost8 = self.trans_conv5(cost7_relu) 

        cost8 = torch.squeeze(cost8,1)

        pred = F.softmax(cost8, dim=1)
        pred = disparityregression(self.maxdisp)(pred)


        return pred
