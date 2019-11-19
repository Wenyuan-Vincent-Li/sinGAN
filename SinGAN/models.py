import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
   
class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc) # 32
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) # ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x) # [1, 3, 26, 26] -> [1, 32, 24, 24]
        x = self.body(x) # -> [1, 32, 18, 18]
        x = self.tail(x) # -> [1, 1, 16, 16]
        return x # Notice that the out put is a feature map not a scalar.


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #ConvBlock(in = 3, out = 32, ker = 3, pad = 0, stride = 1)
        self.body = nn.Sequential()
        # 3-layer conv block TODO: make it as residuel block?
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1) # ConvBlock(in = 32, out = 32, ker = 3, pad = 0, stride = 1)
            self.body.add_module('block%d'%(i+1),block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size), # # ConvBlock(in = 32, out = 3, ker = 3, pad = 0, stride = 1)
            nn.Tanh() # map the output to 0-1
        )
    def forward(self,x,y):
        '''

        :param x: noise layer: z vectors
        :param y: prev image: previous generated images
        :return:
        '''
        # TODO: make generator conditional
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y # Note that the final output is sum
