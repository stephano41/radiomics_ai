import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from .segresnet import SegResNetVAE2
from collections.abc import Sequence
from collections import OrderedDict
import logging
import re
from ..utils import expand_weights

logger = logging.getLogger(__name__)



def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetEncoder(SegResNetVAE2):

    def __init__(self,
                 block,
                 blocks_down,
                 input_image_size: Sequence[int],
                in_channels: int = 1,
                vae_nz=256,
                 shortcut_type='B',
                 no_cuda = False,
                 pretrained_param_path=None,
                 **kwargs):
        self.inplanes = 64
        self.no_cuda = no_cuda

        if isinstance(block, str):
            if block=='BasicBlock':
                block = BasicBlock
            elif block=='Bottleneck':
                block = Bottleneck
            else:
                raise ValueError(f"Block type not implemented, maybe pass the object instead, got {block} as string")

        super().__init__(input_image_size=input_image_size, vae_nz=vae_nz, in_channels=in_channels, blocks_up=blocks_down[:-1],
                         blocks_down=blocks_down, init_filters=64, **kwargs)

        self.convInit = nn.Sequential(OrderedDict([('conv1', nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)), # halves it
            ('bn1', nn.BatchNorm3d(64)),
            ('relu', nn.ReLU(inplace=True))]))
        

        self.down_layers = nn.Sequential(OrderedDict([
            ('down_max_pool', nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)),
            ('layer1', self._make_layer(block, 64, blocks_down[0], shortcut_type, stride=1)),
            ('layer2', self._make_layer(block, 128, blocks_down[1], shortcut_type, stride=2)),
            ('layer3', self._make_layer(block, 256, blocks_down[2], shortcut_type, stride=1, dilation=2)),
            ('layer4', self._make_layer(block, 512, blocks_down[3], shortcut_type, stride=1, dilation=4))
        ]))
    

        self.inplanes=256
        layer3_reverse = self._make_layer(block, 256, blocks_down[2], shortcut_type, stride=1, dilation=2)
        self.inplanes=128
        layer2_reverse = self._make_layer(block, 128, blocks_down[1], shortcut_type, stride=1)
        self.inplanes=64
        layer1_reverse = self._make_layer(block, 64, blocks_down[0], shortcut_type, stride=1)
        
        self.up_layers = nn.Sequential(OrderedDict([
            ('layer3_reverse', layer3_reverse), 
            ('layer2_reverse', layer2_reverse),
            ('layer1_reverse', layer1_reverse)]))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained_param_path is not None:
            self.load_params(pretrained_param_path)
            # freeze the first half of params
            # freezing layers didn't help, actually made it worse
            # self.freeze_params(r'\.layer\d\.')

    def freeze_params(self, regex_pattern):
        for name, param in self.named_parameters():
            if re.search(regex_pattern, name):
                param.requires_grad = False

    def load_params(self, param_path):
        pretrained_weights = torch.load(param_path)['state_dict']

        processed_pretrained_weights = _preprocess_pretrain_weights(pretrained_weights)

        loaded_weights={}
        keys_used=0

        for k,v in self.state_dict().items():
            stripped_key = re.search(r'\.(.*)',k).group()
            if stripped_key in processed_pretrained_weights.keys():
                loaded_weights[k] = expand_weights(processed_pretrained_weights[stripped_key], v)
                keys_used += 1
                logger.debug(f"{k} was loaded as {loaded_weights[k].shape} from {processed_pretrained_weights[stripped_key].shape}")
            else:
                loaded_weights[k] = v
        
        logger.debug(f"{keys_used} keys used out of {len(pretrained_weights.keys())}")
        
        self.load_state_dict(loaded_weights)


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, enable_downsample=True):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion) and enable_downsample:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

def _preprocess_pretrain_weights(pretrained_weights):
    pattern = r'layer\d'

    processed_pretrained_weights = {}
    for k, v in pretrained_weights.items():
        k = k.removeprefix('module')
        processed_pretrained_weights[k] = v
        matched_string = re.search(pattern, k)
        if matched_string is not None:
            processed_pretrained_weights[k.replace(matched_string.group(), matched_string.group()+'_reverse')] = v          
    return processed_pretrained_weights





def med3d_resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetEncoder(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def med3d_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def med3d_resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def med3d_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def med3d_resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def med3d_resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def med3d_resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetEncoder(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model