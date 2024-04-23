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

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


class WideBottleneck(resnet.Bottleneck):
    expansion = 2


def generate_model(model_depth, k, **kwargs):
    assert model_depth in [50, 101, 152, 200]

    inplanes = [x * k for x in resnet.get_inplanes()]
    if model_depth == 50:
        model = resnet.ResNet(WideBottleneck, [3, 4, 6, 3], inplanes, **kwargs)
    elif model_depth == 101:
        model = resnet.ResNet(WideBottleneck, [3, 4, 23, 3], inplanes, **kwargs)
    elif model_depth == 152:
        model = resnet.ResNet(WideBottleneck, [3, 8, 36, 3], inplanes, **kwargs)
    elif model_depth == 200:
        model = resnet.ResNet(WideBottleneck, [3, 24, 36, 3], inplanes,
                              **kwargs)

    return model
