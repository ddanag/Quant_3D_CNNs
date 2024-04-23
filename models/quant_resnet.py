# Copyright (c) 2024 -      Dana Diaconu
# Copyright (c) 2017 - 2020 Kensho Hara

# MIT License
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/diaconu.d/mywork/brevitas_3D_CNN/src/')
import brevitas

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import IntBias, Int8WeightPerTensorFloat
from brevitas.core.restrict_val import RestrictValueType

from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_conv import QuantConv3d
from brevitas.nn.quant_avg_pool import QuantAvgPool3d, QuantAdaptiveAvgPool3d
from brevitas.nn.quant_max_pool import QuantMaxPool3d

class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

FIRST_LAYER_BIT_WIDTH = 8

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, weight_bit_width, stride=1):
    return QuantConv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False,
                     weight_quant=CommonIntWeightPerChannelQuant, 
                     weight_bit_width=weight_bit_width
                     )


def conv1x1x1(in_planes, out_planes, weight_bit_width, stride=1):
    return QuantConv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False,
                     weight_quant=CommonIntWeightPerChannelQuant, 
                     weight_bit_width=weight_bit_width
                     )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, weight_bit_width, act_bit_width, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, weight_bit_width, stride)
        #import pdb; pdb.set_trace()
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = QuantReLU(act_quant=CommonUintActQuant,
                                bit_width=act_bit_width,
                                scaling_per_channel=False,
                                return_quant_tensor=True,
                                inplace=True
                                )
        self.conv2 = conv3x3x3(planes, planes, weight_bit_width)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, in_planes, planes, weight_bit_width, act_bit_width, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes, weight_bit_width)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, weight_bit_width)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, weight_bit_width)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = QuantReLU(act_quant=CommonUintActQuant,
                                bit_width=act_bit_width,
                                inplace=True,
                                scaling_per_channel=False,
                                return_quant_tensor=True)
        self.downsample = downsample
        self.stride = stride

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


class QuantResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 weight_bit_width,
                 act_bit_width,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400
                 ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = QuantConv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False,
                               weight_bit_width=weight_bit_width)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = QuantReLU(act_quant=CommonUintActQuant,
                            bit_width=act_bit_width,
                            inplace=True,
                            scaling_per_channel=False,
                            return_quant_tensor=True)
        self.maxpool = QuantMaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, weight_bit_width, act_bit_width)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       weight_bit_width,
                                       act_bit_width,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       weight_bit_width,
                                       act_bit_width,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       weight_bit_width,
                                       act_bit_width,
                                       stride=2)

        self.avgpool = QuantAdaptiveAvgPool3d((1, 1, 1), bit_width=act_bit_width)
        #import pdb; pdb.set_trace()
        self.fc = QuantLinear(block_inplanes[3] * block.expansion, n_classes, bias=True,
                                bias_quant=IntBias, weight_quant=CommonIntWeightPerTensorQuant,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH)

        for m in self.modules():
            if isinstance(m, QuantConv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, weight_bit_width, act_bit_width, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, weight_bit_width, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  weight_bit_width=weight_bit_width, 
                  act_bit_width=act_bit_width))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, weight_bit_width=weight_bit_width, 
                  act_bit_width=act_bit_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = QuantResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        
        model = QuantResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
        #import pdb; pdb.set_trace()
    elif model_depth == 34:
        model = QuantResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = QuantResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = QuantResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = QuantResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = QuantResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
