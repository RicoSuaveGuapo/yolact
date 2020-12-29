from collections import defaultdict
from itertools import product
from math import sqrt
import math
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import Bottleneck
from torch.cuda.amp import autocast

from backbone import construct_backbone
from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from utils import timer
from utils.functions import MovingAverage, make_net

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
use_jit = torch.cuda.device_count() <= 1
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

# NOTE: I turn off the ScriptModule just for inspectation of the model
# ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
use_jit = False
ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


def outSize(input_size:int, kernal=None, stride=1, padding=0, 
            pool=None, adapool:bool=False, with_pool:bool=False, 
            out_channels:int=None, repeat:int=1, final:bool=False):

    if with_pool:
        assert pool != None, 'if with pooling, must given the pool size.'
        assert repeat >= 1, 'argument repeat only eats >= 1'
    
    if adapool:
        assert type(adapool) == tuple, 'if using adapool, must given the output shape in tuple'
        assert len(adapool) == 2, 'only 2D is supported'
        assert adapool[0] == adapool[1], 'only support equal size AdaptiveAvgPool2d'

    # recall the formula (W−F+2P)/S+1
    if not adapool:
        if not final:
            if repeat == 1:
                if with_pool:
                    outsize = math.floor((input_size - kernal   + 2 * padding)/stride +1)
                    outsize = math.floor((outsize - pool + 2 * padding)/pool +1)
                else:
                    outsize = math.floor((input_size - kernal   + 2 * padding)/stride +1)
            else:
                outsize = input_size
                if with_pool:
                    for i in range(repeat):
                        outsize = math.floor((outsize - kernal + 2 * padding)/stride +1)
                        outsize = math.floor((outsize - pool   + 2 * padding)/pool +1)
                else:
                    for i in range(repeat):
                        outsize = math.floor((outsize - kernal + 2 * padding)/stride +1)
        else:
            outsize = input_size**2*out_channels
    else:
        outsize = adapool[0]
    
    return outsize


class Discriminator(nn.Module):
    def __init__(self, i_size, s_size, in_channels = 3, NUM_CLASSES = 7):
        '''
        # 0 for Fake/Generated
        # 1 for True/Ground Truth
        '''
        # I assume they are the same
        assert i_size == s_size, "image size and segmentation/ground size are not the same"

        super().__init__()
        i_channel = [64,128]
        o_channel = [64,128]
        c_channel = [256,32,8,1]

        self.conv1_i = nn.Conv2d(in_channels=in_channels,  out_channels=i_channel[0], kernel_size=3)
        self.conv2_i = nn.Conv2d(in_channels=i_channel[0], out_channels=i_channel[1], kernel_size=3)
        self.conv1_s = nn.Conv2d(in_channels=NUM_CLASSES,  out_channels=o_channel[0], kernel_size=3)
        self.conv2_s = nn.Conv2d(in_channels=o_channel[0], out_channels=o_channel[1], kernel_size=3)
        self.conv1_c = nn.Conv2d(in_channels=c_channel[0], out_channels=c_channel[1], kernel_size=3)
        self.conv2_c = nn.Conv2d(in_channels=c_channel[1], out_channels=c_channel[2], kernel_size=3)
        self.conv3_c = nn.Conv2d(in_channels=c_channel[2], out_channels=c_channel[3], kernel_size=3)
        
        i_size    = outSize(i_size,3,1,0, pool=3, with_pool=True, repeat = 2)
        # if i_size == s_size then the below is redundant
        # s_size  = outSize(s_size,3,1,0, repeat = 2)
        c_size    = outSize(i_size,3,1,0, repeat = 3)
        feat_size = outSize(c_size, out_channels=c_channel[-1], final=True)
        
        self.dense1 = nn.Linear(in_features=feat_size, out_features=32)
        self.drop   = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(in_features=32, out_features=2)
        
        self.maxpool = nn.MaxPool2d(3)
        self.act     = F.relu
    
    def forward(self, img, seg):
        x1 = img # original image
        x2 = seg # could be ground truth or prediction
        
        # img
        x1 = self.conv1_i(x1)
        x1 = self.act(x1)
        x1 = self.maxpool(x1)
        x1 = self.conv2_i(x1)
        x1 = self.act(x1)
        x1 = self.maxpool(x1)
        
        # seg
        x2 = self.conv1_s(x2)
        x2 = self.act(x2)
        x2 = self.maxpool(x2)
        x2 = self.conv2_s(x2)
        x2 = self.act(x2)
        x2 = self.maxpool(x2)
        
        concat = torch.cat([x1, x2], dim=1)
        
        c = self.conv1_c(concat)
        c = self.act(c)
        c = self.conv2_c(c)
        c = self.act(c)
        c = self.conv3_c(c)
        c = self.act(c)
        
        c = c.view(c.shape[0], -1)
        c = self.dense1(c)
        c = self.act(c)
        c = self.drop(c)
        c = self.dense2(c)
        
        return c


class Discriminator_StandFord(nn.Module):
    def __init__(self, i_size, s_size, num_classes=6, in_channels = 3):

        '''
        0 for Fake/Generated
        1 for True/Ground Truth

        num_classes: number of classes (for seg branch)
        in_channels: number of channels (for original image)
        '''
        # I assume they are the same
        assert i_size == s_size, "image size and segmentation/ground size are not the same"

        super().__init__()
        i_channel = [64]
        s_channel = [64]
        c_channel = [128,256,512,1024,1]
        i_kernel  = [5]
        c_kernel  = [3,3,3,3,3]

        self.conv1_i = nn.Conv2d(in_channels=in_channels,  out_channels=i_channel[0], kernel_size=i_kernel[0])
        self.conv1_s = nn.Conv2d(in_channels=num_classes,  out_channels=s_channel[0], kernel_size=i_kernel[0])

        self.conv1_c = nn.Conv2d(in_channels=i_channel[0] + s_channel[0], out_channels=c_channel[0], kernel_size=c_kernel[0])
        self.conv2_c = nn.Conv2d(in_channels=c_channel[0], out_channels=c_channel[1], kernel_size=c_kernel[1])
        self.conv3_c = nn.Conv2d(in_channels=c_channel[1], out_channels=c_channel[2], kernel_size=c_kernel[2])
        self.conv4_c = nn.Conv2d(in_channels=c_channel[2], out_channels=c_channel[3], kernel_size=c_kernel[3])
        self.conv5_c = nn.Conv2d(in_channels=c_channel[3], out_channels=c_channel[4], kernel_size=c_kernel[4])

        self.bni     = nn.BatchNorm2d(num_features=i_channel[0])
        self.bns     = nn.BatchNorm2d(num_features=s_channel[0])

        self.bn1c    = nn.BatchNorm2d(num_features=c_channel[0])
        self.bn2c    = nn.BatchNorm2d(num_features=c_channel[1])
        self.bn3c    = nn.BatchNorm2d(num_features=c_channel[2])
        self.bn4c    = nn.BatchNorm2d(num_features=c_channel[3])
        self.bn5c    = nn.BatchNorm2d(num_features=c_channel[4])

        self.maxpool = nn.MaxPool2d(3)
        self.Adapool = nn.AdaptiveAvgPool2d((3,3))
        self.act     = F.relu
        # Guruntee the output is all positive
        self.finalact= torch.sigmoid
        # self.finalact= nn.Softplus()
    
    def forward(self, img, seg):
        x1 = img # original image
        x2 = seg # could be ground truth or prediction
        
        # img
        x1 = self.bni(self.conv1_i(x1))
        x1 = self.act(x1)
        
        # seg
        x2 = self.bns(self.conv1_s(x2))
        x2 = self.act(x2)
        
        concat = torch.cat([x1, x2], dim=1)
        
        # c = self.act(self.bn1c(self.conv1_c(concat)))
        # c = self.act(self.bn2c(self.conv2_c(c)))
        # c = self.act(self.bn3c(self.conv3_c(c)))
        # c = self.act(self.bn4c(self.conv4_c(c)))


        c = self.act(self.conv1_c(concat))
        c = self.act(self.conv2_c(c))
        c = self.act(self.conv3_c(c))
        c = self.act(self.conv4_c(c))

        c = self.Adapool(c)
        # c = self.bn5c(self.conv5_c(c))
        c = self.conv5_c(c)
        # c = self.act(c)
        c = self.finalact(c)
        c = c.squeeze()
        
        return c

class Discriminator_Dcgan(nn.Module):
    def __init__(self, i_size, s_size, num_classes=6, in_channels = 3):

        '''
        0 for Fake/Generated
        1 for True/Ground Truth

        num_classes: number of classes (for seg branch)
        in_channels: number of channels (for original image)
        '''
        # I assume they are the same
        assert i_size == s_size, "image size and segmentation/ground size are not the same"

        super().__init__()
        i_channel = [64]
        s_channel = [64]
        c_channel = [128,256,512,1024,1]
        i_kernel  = [5]
        c_kernel  = [3,3,3,3,3]

        self.conv1_i = nn.Conv2d(in_channels=in_channels,  out_channels=i_channel[0], kernel_size=i_kernel[0], bias=False)
        self.conv1_s = nn.Conv2d(in_channels=num_classes,  out_channels=s_channel[0], kernel_size=i_kernel[0], bias=False)

        self.conv1_c = nn.Conv2d(in_channels=i_channel[0] + s_channel[0], out_channels=c_channel[0], kernel_size=c_kernel[0], bias=False)
        self.conv2_c = nn.Conv2d(in_channels=c_channel[0], out_channels=c_channel[1], kernel_size=c_kernel[1], bias=False)
        self.conv3_c = nn.Conv2d(in_channels=c_channel[1], out_channels=c_channel[2], kernel_size=c_kernel[2], bias=False)
        self.conv4_c = nn.Conv2d(in_channels=c_channel[2], out_channels=c_channel[3], kernel_size=c_kernel[3], bias=False)
        self.conv5_c = nn.Conv2d(in_channels=c_channel[3], out_channels=c_channel[4], kernel_size=c_kernel[4], bias=False)

        self.bni     = nn.BatchNorm2d(num_features=i_channel[0])
        self.bns     = nn.BatchNorm2d(num_features=s_channel[0])

        self.bn1c    = nn.BatchNorm2d(num_features=c_channel[0])
        self.bn2c    = nn.BatchNorm2d(num_features=c_channel[1])
        self.bn3c    = nn.BatchNorm2d(num_features=c_channel[2])
        self.bn4c    = nn.BatchNorm2d(num_features=c_channel[3])
        self.bn5c    = nn.BatchNorm2d(num_features=c_channel[4])

        self.maxpool = nn.MaxPool2d(3)
        self.Adapool = nn.AdaptiveAvgPool2d((3,3))
        # NOTE
        self.act     = nn.LeakyReLU(0.2, inplace=True)
        # Guruntee the output is all positive
        self.finalact= nn.Sigmoid()
    
    def forward(self, img, seg):
        x1 = img # original image
        x2 = seg # could be ground truth or prediction
        
        # img
        x1 = self.bni(self.conv1_i(x1))
        x1 = self.act(x1)
        
        # seg
        x2 = self.bns(self.conv1_s(x2))
        x2 = self.act(x2)
        
        concat = torch.cat([x1, x2], dim=1)
        
        c = self.act(self.bn1c(self.conv1_c(concat)))
        c = self.act(self.bn2c(self.conv2_c(c)))
        c = self.act(self.bn3c(self.conv3_c(c)))
        c = self.act(self.bn4c(self.conv4_c(c)))

        c = self.Adapool(c)
        # c = self.bn5c(self.conv5_c(c))
        c = self.conv5_c(c)
        # c = self.act(c)
        c = self.finalact(c)
        c = c.squeeze()
        
        return c


class Discriminator_Wgan(nn.Module):
    def __init__(self, i_size, s_size, num_classes=1, in_channels = 3):

        '''
        0 for Fake/Generated
        1 for True/Ground Truth

        num_classes: number of classes (for seg branch)
        in_channels: number of channels (for original image)
        '''
        # I assume they are the same
        assert i_size == s_size, "image size and segmentation/ground size are not the same"
        # the default input size is 1*138*138

        super().__init__()
        i_channel = [64]
        s_channel = [64]
        c_channel = [128,256,512,1024,1]
        i_kernel  = [5]
        c_kernel  = [3,3,3,3,3]

        self.conv1_i = nn.Conv2d(in_channels=in_channels,  out_channels=i_channel[0], kernel_size=i_kernel[0], bias=False)
        self.conv1_s = nn.Conv2d(in_channels=num_classes,  out_channels=s_channel[0], kernel_size=i_kernel[0], bias=False)

        # self.conv1_c = nn.Conv2d(in_channels=3, out_channels=c_channel[0], kernel_size=c_kernel[0], bias=False)
        self.conv1_c = nn.Conv2d(in_channels=i_channel[0] + s_channel[0], out_channels=c_channel[0], kernel_size=c_kernel[0], bias=False)
        self.conv2_c = nn.Conv2d(in_channels=c_channel[0], out_channels=c_channel[1], kernel_size=c_kernel[1], bias=False)
        self.conv3_c = nn.Conv2d(in_channels=c_channel[1], out_channels=c_channel[2], kernel_size=c_kernel[2], bias=False)
        self.conv4_c = nn.Conv2d(in_channels=c_channel[2], out_channels=c_channel[3], kernel_size=c_kernel[3], bias=False)
        self.conv5_c = nn.Conv2d(in_channels=c_channel[3], out_channels=c_channel[4], kernel_size=c_kernel[4], bias=False)
        # self.conv5_c = nn.Conv2d(in_channels=c_channel[1], out_channels=c_channel[4], kernel_size=c_kernel[4], bias=False)

        self.bni     = nn.BatchNorm2d(num_features=i_channel[0])
        self.bns     = nn.BatchNorm2d(num_features=s_channel[0])

        self.bn1c    = nn.BatchNorm2d(num_features=c_channel[0])
        self.bn2c    = nn.BatchNorm2d(num_features=c_channel[1])
        self.bn3c    = nn.BatchNorm2d(num_features=c_channel[2])
        self.bn4c    = nn.BatchNorm2d(num_features=c_channel[3])
        self.bn5c    = nn.BatchNorm2d(num_features=c_channel[4])

        # FIXME: This step may lose a lot of info
        self.Adapool = nn.AdaptiveAvgPool2d((3,3))
        # NOTE
        self.act     = nn.LeakyReLU(0.2, inplace=True)
        # take the advice from WGAN
        # self.finalact= nn.Sigmoid()

        # Taken from gan_mask_rcnn
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
    
    def forward(self, img, seg):
        x1 = img # original image
        x2 = seg # could be ground truth or prediction
        
        # elem_mul = x1*x2

        # img
        x1 = self.bni(self.conv1_i(x1))
        x1 = self.act(x1)
        
        # seg
        x2 = self.bns(self.conv1_s(x2))
        x2 = self.act(x2)
        
        # NOTE
        # elem_mul = x1*x2
        concat = torch.cat([x1, x2], dim=1)
        
        c = self.act(self.bn1c(self.conv1_c(concat)))
        # c = self.act(self.bn1c(self.conv1_c(elem_mul)))
        c = self.act(self.bn1c(self.conv1_c(c)))
        c = self.act(self.bn2c(self.conv2_c(c)))
        c = self.act(self.bn3c(self.conv3_c(c)))
        c = self.act(self.bn4c(self.conv4_c(c)))

        c = self.Adapool(c)
        c = self.conv5_c(c)
        # c = self.finalact(c)
        c = c.squeeze()
        
        # return elem_mul, c
        return c


class Discriminator_MRCNNgan(nn.Module):
    def __init__(self, i_size, s_size, num_classes=1, in_channels = 3):

        '''
        0 for Fake/Generated
        1 for True/Ground Truth

        num_classes: number of classes (for seg branch)
        in_channels: number of channels (for original image)
        '''
        # I assume they are the same
        assert i_size == s_size, "image size and segmentation/ground size are not the same"
        # the default input size is 1*138*138

        super().__init__()
        i_channel = [64]
        s_channel = [64]
        c_channel = [128,256,512,1024]

        i_kernel  = [5]
        i_stride  = [3]
        c_kernel  = [4,4,4,3]
        c_stride  = [2,2,2,2]

        self.conv1_i = nn.Conv2d(in_channels,  i_channel[0], i_kernel[0], i_stride[0], bias=False)
        self.conv1_s = nn.Conv2d(num_classes,  s_channel[0], i_kernel[0], i_stride[0], bias=False)

        self.conv1_c = nn.Conv2d(i_channel[0], c_channel[0], c_kernel[0], c_stride[0], bias=False)
        self.conv2_c = nn.Conv2d(c_channel[0], c_channel[1], c_kernel[1], c_stride[1], bias=False)
        self.conv3_c = nn.Conv2d(c_channel[1], c_channel[2], c_kernel[2], c_stride[2], bias=False)
        self.conv4_c = nn.Conv2d(c_channel[2], c_channel[3], c_kernel[3], c_stride[3], bias=False)

        self.bni     = nn.BatchNorm2d(num_features=i_channel[0])
        self.bns     = nn.BatchNorm2d(num_features=s_channel[0])

        self.bn1c    = nn.BatchNorm2d(num_features=c_channel[0])
        self.bn2c    = nn.BatchNorm2d(num_features=c_channel[1])
        self.bn3c    = nn.BatchNorm2d(num_features=c_channel[2])
        self.bn4c    = nn.BatchNorm2d(num_features=c_channel[3])

        self.act     = nn.LeakyReLU(0.2, inplace=True)

        # Taken from gan_mask_rcnn
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
    # NOTE: for AMP
    @autocast()
    def forward(self, img, seg):
        batch = img.size(0)
        x1 = img # original image
        x2 = seg # could be ground truth or prediction
        # img
        x1 = self.act(self.bni(self.conv1_i(x1)))
        # seg
        x2 = self.act(self.bns(self.conv1_s(x2)))

        # elementwise multiplication
        elem_mul = x1*x2
        # print(elem_mul.size())
        c1 = self.act(self.bn1c(self.conv1_c(elem_mul)))
        # print(c1.size())
        c2 = self.act(self.bn2c(self.conv2_c(c1)))
        # print(c2.size())
        c3 = self.act(self.bn3c(self.conv3_c(c2)))
        # print(c3.size())
        c4 = self.act(self.bn4c(self.conv4_c(c3)))
        # print(c4.size())
        # the coefficients here are becase deeper the layer, the fewer size of feature map
        # the coefficients can help deeper layer be noted.
        output = torch.cat((elem_mul.view(batch,-1), 1*c1.view(batch,-1),\
                            2*c2.view(batch,-1), 3*c3.view(batch,-1), 4*c4.view(batch,-1)),1)
        # torch.Size([2, 212416])

        return output


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)

prior_cache = defaultdict(lambda: None)

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0, torchout=False):
        super().__init__()

        # NOTE: for visualization of model
        self.torchout    = torchout

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim # Defined by Yolact
        self.num_priors  = sum(len(x)*len(scales) for x in aspect_ratios)
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)

            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)
            
            if cfg.use_mask_scoring:
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0]
        
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        
        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)

        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        
        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_mask_scoring:
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)    

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)

                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)
        
        priors = self.make_priors(conv_h, conv_w, x.device)

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        if cfg.use_mask_scoring:
            preds['score'] = score

        if cfg.use_instance_coeff:
            preds['inst'] = inst

        if self.torchout:
            preds = [v for k, v in preds.items()]
        
        return preds

    def make_priors(self, conv_h, conv_w, device):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        global prior_cache
        size = (conv_h, conv_w)

        with timer.env('makepriors'):
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for ars in self.aspect_ratios:
                        for scale in self.scales:
                            for ar in ars:
                                if not cfg.backbone.preapply_sqrt:
                                    ar = sqrt(ar)

                                if cfg.backbone.use_pixel_scales:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                                else:
                                    w = scale * ar / conv_w
                                    h = scale / ar / conv_h
                                
                                # This is for backward compatability with a bug where I made everything square by accident
                                if cfg.backbone.use_square_anchors:
                                    h = w

                                prior_data += [x, y, w, h]

                self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach()
                self.priors.requires_grad = False
                self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
                self.last_conv_size = (conv_w, conv_h)
                prior_cache[size] = None
            elif self.priors.device != device:
                # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
                if prior_cache[size] is None:
                    prior_cache[size] = {}
                
                if device not in prior_cache[size]:
                    prior_cache[size][device] = self.priors.to(device)

                self.priors = prior_cache[size][device]
        
        return self.priors

class FPN(ScriptModuleWrapper):
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers.
        - interpolation_mode (str): The mode to pass to F.interpolate.
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.
                                These extra layers are downsampled from the last selected layer.

    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers',
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):
        super().__init__()

        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability
        padding = 1 if cfg.fpn.pad else 0
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)
            for _ in in_channels
        ])

        if cfg.fpn.use_conv_downsample:
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode
        self.num_downsample         = cfg.fpn.num_downsample
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels.
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device)
        for i in range(len(convouts)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts) # 3 in base config
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x
        
        # This janky second loop is here because TorchScript.
        j = len(convouts)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = pred_layer(out[j])

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True)

        cur_idx = len(out)

        # In the original paper, this takes care of P6
        if self.use_conv_downsample: # True in base config
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))

        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out

class FastMaskIoUNet(ScriptModuleWrapper):

    def __init__(self):
        super().__init__()
        input_channels = 1
        last_layer = [(cfg.num_classes-1, 1, {})]
        self.maskiou_net, _ = make_net(input_channels, cfg.maskiou_net + last_layer, include_last_relu=True)

    def forward(self, x):
        x = self.maskiou_net(x)
        maskiou_p = F.max_pool2d(x, kernel_size=x.size()[2:]).squeeze(-1).squeeze(-1)

        return maskiou_p



class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, test_code=False, torchout=False):
        super().__init__()
        self.torchout = torchout
        self.test_code = test_code
        self.backbone = construct_backbone(cfg.backbone)

        if cfg.freeze_bn:
            self.freeze_bn()

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            # The input layer to the mask prototype generation network
            self.proto_src = cfg.mask_proto_src
            
            if self.proto_src is None: in_channels = 3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features # 256
            else: in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1


        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels

        if cfg.use_maskiou:
            self.maskiou_net = FastMaskIoUNet()

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)


        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)

        for idx, layer_idx in enumerate(self.selected_layers):
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)
            self.prediction_layers.append(pred)

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')
        
        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                        all_in(module.__dict__['_constants_set'], conv_constants)
                        and all_in(conv_constants, module.__dict__['_constants_set']))
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
    
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
    
    # NOTE: for AMP
    @autocast()
    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w
        
        # ---------------- backbone ---------------- #
        with timer.env('backbone'):
            outs = self.backbone(x)


        # ----------------   FPN   ----------------- #
        if cfg.fpn is not None:
            with timer.env('fpn'):
                # Use backbone.selected_layers because we overwrote self.selected_layers
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)

        # ---------------- protoNet ---------------- #
        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)

                if cfg.mask_proto_prototypes_as_features:
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        # ---------------- pred_heads ---------------- #
        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if cfg.use_mask_scoring:
                pred_outs['score'] = []

            if cfg.use_instance_coeff:
                pred_outs['inst'] = []
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                # outputs from FPN
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                # A hack for the way dataparallel works
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                    pred_layer.parent = [self.prediction_layers[0]]
                
                # NOTE:
                # the outputs from FPN goes into the prediction head
                p = pred_layer(pred_x)
                # get the output in the format:
                # p = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
                # Input
                #     - x: The convOut from a layer in the backbone network
                #         Size: [batch_size, in_channels, conv_h, conv_w])

                # Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
                #     - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
                #     - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
                #     - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
                #     - prior_boxes: [conv_h*conv_w*num_priors, 4]
                # put them into the pred_outs
                for k, v in p.items():
                    pred_outs[k].append(v)
                    
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)

        if proto_out is not None:
            pred_outs['proto'] = proto_out

        # ----- Training output -----
        if self.training:
            # For the extra loss functions
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            if self.torchout:
                pred_head = [k for k, v in pred_outs.items()]
                pred_outs = [v for k, v in pred_outs.items()]
                return (pred_head, pred_outs)
            else:
                return pred_outs

        # NOTE
        # -----  GAN evalation  -----
        elif cfg.pred_seg and cfg.gan_eval:
            pred_outs['segm'] = self.semantic_seg_conv(outs[0])
            return pred_outs

        # ----- Validation/Test -----
        else:
            if cfg.use_mask_scoring:
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            if cfg.use_focal_loss:
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            else:

                if cfg.use_objectness_score:
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                        * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            
            if self.torchout:
                pred_head = [k for k, v in pred_outs.items()]
                pred_outs = [v for k, v in pred_outs.items()]
                return (pred_head, pred_outs)
            else:
                if not self.test_code:
                    return self.detect(pred_outs, self)
                else:
                    return self.detect(pred_outs, self)[0]['detection']['box']


# Some testing code
if __name__ == '__main__':
    # from utils.functions import init_console
    # init_console()

    # # Use the first argument to set the config if you want
    # import sys
    # if len(sys.argv) > 1:
    #     from data.config import set_cfg
    #     set_cfg(sys.argv[1])

    # net = Yolact()
    # net.train()
    # net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # # GPU
    # net = net.cuda()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
    # y = net(x)

    # for p in net.prediction_layers:
    #     print(p.last_conv_size)

    # print()
    # for k, a in y.items():
    #     print(k + ': ', a.size(), torch.sum(a))
    # exit()
    
    # net(x)
    # # timer.disable('pass2')
    # avg = MovingAverage()
    # try:
    #     while True:
    #         timer.reset()
    #         with timer.env('everything else'):
    #             net(x)
    #         avg.add(timer.total_time())
    #         print('\033[2J') # Moves console cursor to 0,0
    #         timer.print_stats()
    #         print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    # except KeyboardInterrupt:
    #     pass

    # NOTE
    # ----- Rico Test Section -----
    import warnings

    import netron
    import torch.utils.data as data
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from data import *
    from data import MEANS, STD, cfg
    from data.coco import detection_collate
    from layers.modules import MultiBoxLoss
    from train import *
    from utils.augmentations import *
    from utils.augmentations import BaseTransform, SSDAugmentation
    warnings.filterwarnings("ignore")

    # ----- Dataset Inspectation -----
    # dataset = COCODetection(image_path=cfg.dataset.train_images,
    #                         info_file=cfg.dataset.train_info,
    #                         transform=SSDAugmentation(MEANS))
    # ----- output format ----- 
    # >>> im, (gt, masks, num_crowds)
    # data_loader = data.DataLoader(dataset,batch_size=2,shuffle=False,collate_fn=detection_collate,pin_memory=True)
    # detection_collate
    # Custom collate fn for dealing with batches of images that have a different
    # number of associated object annotations (bounding boxes).
    # 1) (tensor) batch of images stacked on their 0 dim
    # 2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
    #     on 0 dim. The output gt is a tuple of annotations and masks.
    #   imgs, (targets, masks, num_crowds)
    # datum = next(iter(data_loader))
    # Because in the discriminator training, we do not 
    # want the gradient flow back to the generator part
    # we detach datum. Turn out that we can just modify the
    # Netloss
    # detatch_datum = []
    # for i, data in enumerate(datum):
    #     if i == 0:
    #         imgs = data
    #         imgs = [img.detach() for img in imgs]
    #         detatch_datum.append(imgs)
    #     else:
    #         (targets, masks, num_crowds) = data
    #         targets   = [target.detach() for target in targets]
    #         masks     = [mask.detach() for mask in masks]
    #         new_datum = (targets, masks, num_crowds)
    #         detatch_datum.append(new_datum)

    # detatch_datum = tuple(detatch_datum)
    
    # print(len(datum))
    # >>> 2
    # ----- img ----- 
    # show the batch
    # print(len(datum[0]))
    # show image size
    # print(datum[0][0].size())
    # >>> torch.Size([3, 550, 550])

    # ----- Target format ----- 
    # show the batch
    # print(len(datum[1][0]))
    # show the targets size
    # print(datum[1][0][0].size())
    # >>> torch.Size([2, 5])
    # gt number, (x, y, h, w, label)
    
    # ----- Mask format ----- 
    # show batch
    # print(len(datum[1][1]))
    # While training
    # print(datum[1][1][0].size())
    # >>> torch.Size([3, 550, 550])
    # gt number, h, w
    # ----- If validation ----- 
    # >>> Mask format is original image format
    
    # ----- Example of input ----- 
    # img_np = img[1][1][0].numpy()
    # img_np = img_np.transpose(2,1,0)*200
    # cv2.imwrite('/home/rico-li/Job/豐興鋼鐵/EDA/img_np.jpg', img_np) 
    # print(img[1][2])
    # img = cv2.imread('/home/rico-li/Job/豐興鋼鐵/data/clean_data_20frames/yolact_train/JPEGImages/mod_1500_curve_3_frame0551.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (550, 550))
    # mean = np.array(MEANS)
    # std  = np.array(STD)
    # img = (img - mean)/std
    # img = img.transpose(2,0,1)
    # img = torch.from_numpy(img)
    # img = img.unsqueeze(0).cuda().float()

    # # ----- Net Inspectation ----- 
    # net = Yolact(torchout=False)
    # net.load_weights('weights/yolact_base_1249_60000.pth')
    # net.training = True
    # preds = net(img)
    # print(type(preds['loc'])
    # pred_head, pred_outs = net(img)
    # print(pred_head)
    # # >>> ['loc', 'conf', 'mask', 'priors', 'proto']
    # print([i.size() for i in pred_outs])

    # ----- Check the process in the trainning -----
    # net = Yolact(test_code=False,torchout=False)
    # net.load_weights('weights/yolact_base_1249_60000.pth')
    # net.training = True
    # ----- Loss Inspectation -----
    # pred_seg = True

    # cfg.dataset = metal2020_dataset
    # cfg.config  = yolact_base_config
    # criterion = MultiBoxLoss(num_classes=cfg.num_classes,
    #                          pos_threshold=cfg.positive_iou_threshold,
    #                          neg_threshold=cfg.negative_iou_threshold,
    #                          negpos_ratio=cfg.ohem_negpos_ratio, pred_seg=pred_seg)
    # net = CustomDataParallel(NetLoss(net, criterion, pred_seg=pred_seg))
    # args.batch_alloc = [1]
    # dataset = COCODetection(image_path=cfg.dataset.train_images,
    #                         info_file=cfg.dataset.train_info,
    #                         transform=SSDAugmentation(MEANS))
    # data_loader = data.DataLoader(dataset, batch_size=1,
    #                               shuffle=False, collate_fn=detection_collate)
    # datum = next(iter(data_loader))
    # losses = net(datum)
    # Ground truth Mask 
    # >>> datum[1][1][0]
    # if pred_seg:
    #     losses, pred_seg_list, label_t_list = net(datum)
    #     pred_seg_list = [torch.clamp(v.permute(2,1,0).contiguous(), 0, 1) for v in pred_seg_list]
    #     b = len(pred_seg_list) # batch size
    #     # print([i.size() for i in pred_seg_list])
    #     # print([i.size() for i in label_t_list])
    #     _, h, w = pred_seg_list[0].size()
    #     # neglact the background class
    #     pred_seg_clas = torch.zeros(b, cfg.num_classes-1, h, w)
    #     for idx in range(b):
    #         for i, label in enumerate(label_t_list[idx]):
    #             pred_seg_clas[idx, label, ...] += pred_seg_list[idx][i,...]

    #     print(pred_seg_clas[0])
    #     print([pred_seg_clas[0][i] for i in label_t_list[0]])

        # npimg = pred_seg_list[0][0,...].detach().cpu().numpy()*255
        # from PIL import Image
        # im = Image.fromarray(npimg)
        # im = im.convert('L')
        # im.save('pred_seg_list.jpg')

    # else:
    #     losses = net(datum)
    #     print(losses)
    # ----- output format -----
    # >>> {'B': tensor(0.1974, grad_fn=<DivBackward0>), 'M': tensor(0.2925, grad_fn=<DivBackward0>), 
    # >>> 'C': tensor(2.8976, grad_fn=<DivBackward0>), 'S': tensor(0.0147, grad_fn=<DivBackward0>)}
    # Loss Key:
    #  - B: Box Localization Loss
    #  - C: Class Confidence Loss
    #  - M: Mask Loss
    #  - P: Prototype Loss
    #  - D: Coefficient Diversity Loss
    #  - E: Class Existence Loss
    #  - S: Semantic Segmentation Loss

    # ----- Check the process in the detection -----
    # net = Yolact(test_code=False,torchout=False)
    # net.load_weights('weights/yolact_base_1249_60000.pth')
    # net.training = False

    # cfg.dataset = metal2020_dataset
    # cfg.config  = yolact_base_config
    # criterion = MultiBoxLoss(num_classes=cfg.num_classes,
    #                          pos_threshold=cfg.positive_iou_threshold,
    #                          neg_threshold=cfg.negative_iou_threshold,
    #                          negpos_ratio=cfg.ohem_negpos_ratio)
    # net = CustomDataParallel(NetLoss(net, criterion))
    # args.batch_alloc = [1]
    # dataset = COCODetection(image_path=cfg.dataset.valid_images,
    #                         info_file=cfg.dataset.valid_info,
    #                         transform=BaseTransform(MEANS))
    # data_loader = data.DataLoader(dataset, batch_size=1,
    #                               shuffle=False, collate_fn=detection_collate)
    # datum = next(iter(data_loader))
    # output = net(datum)
    # ----- Yolact net output format -----
    # >>> (batch, {'detection':, 'net':})
    # ----- detection format -----
    # >>> {'box':, 'mask':, 'class':, 'score':, 'proto':]}


    # fakeimg1 = torch.Tensor(1,512,128,128)
    # fakeimg2 = torch.Tensor(1,1024,128,128)
    # fakeimg3 = torch.Tensor(1,2048,128,128)
    # fpn = FPN([512,1024,2048])
    # fpnout = fpn([fakeimg1, fakeimg2, fakeimg3])
    # torch.onnx.export(fpn, [fakeimg1, fakeimg2, fakeimg3], 'runs/fpn.onnx')
    # netron.start('runs/fpn.onnx')

    # num_grids = 0
    # in_channels = 3 + num_grids
    # proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)
    # proto_x = img
    # proto_out = proto_net(proto_x)
    # torch.onnx.export(proto_net, proto_x, 'runs/proto_net.onnx')
    # netron.start('runs/proto_net.onnx')

    # _, _, cfg._tmp_img_h, cfg._tmp_img_w = fakeimg1.size()
    # src_channels  = 256
    # cfg.num_heads = 3
    # cfg.mask_dim  = 550
    # pred = PredictionModule(src_channels, src_channels,
    #                                 aspect_ratios = cfg.backbone.pred_aspect_ratios[0],
    #                                 scales        = cfg.backbone.pred_scales[0],
    #                                 parent        = None,
    #                                 index         = 0,
    #                                 torchout=True)
    # # pout = pred(fpnout[0])
    
    # torch.onnx.export(pred, fpnout[0], 'runs/pred.onnx')
    # netron.start('runs/pred.onnx')


    # img = torch.zeros([2,3,550,550])
    # seg = torch.zeros([2,7,550,550])
    # _, in_channels, i_size, _ = img.size()
    # _, NUM_CLASSES, s_size, _ = seg.size()
    # discriminator = Discriminator(in_channels = in_channels, NUM_CLASSES = NUM_CLASSES, i_size=i_size, s_size=s_size)
    # c = discriminator(img, seg)

    # --- GAN evaluation ---
    # Note that the cfg.pred_seg must be True for the following
    # cfg.gan_eval = True
    # net = Yolact(torchout=False)
    # net.load_weights('weights/yolact_base_1249_60000.pth')
    # net.eval()
    # pred_seg = True

    # cfg.dataset = metal2020_dataset
    # cfg.config  = yolact_base_config
    # criterion = MultiBoxLoss(num_classes=cfg.num_classes,
    #                          pos_threshold=cfg.positive_iou_threshold,
    #                          neg_threshold=cfg.negative_iou_threshold,
    #                          negpos_ratio=cfg.ohem_negpos_ratio, pred_seg=pred_seg)
    # net = CustomDataParallel(NetLoss(net, criterion, pred_seg=pred_seg))
    # args.batch_alloc = [3]
    # val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
    #                                 info_file=cfg.dataset.valid_info,
    #                                 transform=BaseTransform(MEANS))
    # val_loader  = data.DataLoader(val_dataset, batch_size=3,
    #                               num_workers=12*2,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)
    # for datum in tqdm(val_loader, desc='GAN Validation'):
    #     with torch.no_grad():
    #         losses, seg_list, pred_list = net(datum)

    # for val_i in range(len(val_dataset)):
    #     img, gt, gt_masks, h, w, num_crowd = val_dataset.pull_item(val_i)
    #     batch = img.unsqueeze(0).cuda()
    #     with torch.no_grad():
    #         preds = net(batch)
    #         print(preds[0].keys())
    #         break

    img = torch.randn([2,3,138,138])
    seg = torch.randn([2,1,138,138])
    dis_net = Discriminator_MRCNNgan(i_size=138, s_size=138)
    dis_net(img, seg)

