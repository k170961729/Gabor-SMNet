import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--datapath', default='/home/zhendong/dataset/SceneFlow/', help='datapath')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--loadmodel', default= None, help='load model')
parser.add_argument('--savemodel', default='./trained/', help='save model')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size= 4, shuffle= True, num_workers= 24, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size= 2, shuffle= False, num_workers= 12, drop_last=False)


model = stackhourglass(args.maxdisp)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])


# p.data.nelement 返回p中元素的个数
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

#优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    #---------
    mask = disp_true < args.maxdisp  
    mask.detach_()  #return a new Tensor, detached from the current graph, the result will never require gradient.
    #----
    optimizer.zero_grad()


    output = model(imgL,imgR)
    output = torch.squeeze(output, 1)   #return a tensor with all the dimensions of input of size 1 removed
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()  
    optimizer.step()  

    return loss.item()


def test(imgL,imgR,disp_true):

    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    #---------
    mask = disp_true < 192
    #----

    with torch.no_grad():
        output = model(imgL,imgR)

    output = torch.squeeze(output.data.cpu(),1)[:,4:,:]

    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_full_time = time.time()
    for epoch in range(1, args.epochs + 1):

        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            # channel, height, width
            loss = train(imgL_crop, imgR_crop, disp_crop_L)

            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader)}, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))


    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):

        test_loss = test(imgL,imgR, disp_L)

        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))

        total_test_loss += test_loss

    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.savemodel+'testinformation.tar'

    torch.save({'test_loss': total_test_loss/len(TestImgLoader)}, savefilename)


if __name__ == '__main__':

   main()
