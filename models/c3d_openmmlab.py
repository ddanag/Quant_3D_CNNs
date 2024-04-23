# Copyright (c) 2024 -      Dana Diaconu
# Copyright (c) 2019 - 2023 Hao Ren (leftthomas) 
# Original source: https://github.com/leftthomas/R2Plus1D-C3D

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
from collections import OrderedDict

class C3D(nn.Module):
    """
    The C3D network as described in
    Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
    Proceedings of the IEEE international conference on computer vision. 2015.
    """
    # Modifying this to try to reach a good accuracy as close as possible to MMAction2.

    def __init__(self, num_classes, input_channel=3):
        super(C3D, self).__init__()

        
        self.conv1a = nn.Conv3d(input_channel, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(4096, num_classes)

        self.__init_weight()

    def forward(self, x):

        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        logits = self.fc_cls(x)

        return logits
        #return x

    def __init_weight(self):
        pretrained = True
        if pretrained:
            model = torch.load('models/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth')
            k = model['state_dict'].keys()
            self.conv1a.weight = nn.Parameter(model['state_dict']['backbone.conv1a.conv.weight'])
            #import pdb; pdb.set_trace()
            self.conv1a.bias = nn.Parameter(model['state_dict']['backbone.conv1a.conv.bias'])
            self.conv2a.weight = nn.Parameter(model['state_dict']['backbone.conv2a.conv.weight'])
            self.conv2a.bias = nn.Parameter(model['state_dict']['backbone.conv2a.conv.bias'])            
            self.conv3a.weight = nn.Parameter(model['state_dict']['backbone.conv3a.conv.weight'])
            self.conv3a.bias = nn.Parameter(model['state_dict']['backbone.conv3a.conv.bias'])           
            self.conv3b.weight = nn.Parameter(model['state_dict']['backbone.conv3b.conv.weight'])
            self.conv3b.bias = nn.Parameter(model['state_dict']['backbone.conv3b.conv.bias'])
            self.conv4a.weight = nn.Parameter(model['state_dict']['backbone.conv4a.conv.weight'])
            self.conv4a.bias = nn.Parameter(model['state_dict']['backbone.conv4a.conv.bias'])         
            self.conv4b.weight = nn.Parameter(model['state_dict']['backbone.conv4b.conv.weight'])
            self.conv4b.bias = nn.Parameter(model['state_dict']['backbone.conv4b.conv.bias'])   
            self.conv5a.weight = nn.Parameter(model['state_dict']['backbone.conv5a.conv.weight'])
            self.conv5a.bias = nn.Parameter(model['state_dict']['backbone.conv5a.conv.bias'])         
            self.conv5b.weight = nn.Parameter(model['state_dict']['backbone.conv5b.conv.weight'])
            self.conv5b.bias = nn.Parameter(model['state_dict']['backbone.conv5b.conv.bias'])   
            
            #self.fc6.weight = nn.Parameter(model['state_dict']['backbone.fc6.weight'])
            #self.fc6.bias = nn.Parameter(model['state_dict']['backbone.fc6.bias'])
            #self.fc7.weight = nn.Parameter(model['state_dict']['backbone.fc7.weight'])
            #self.fc7.bias = nn.Parameter(model['state_dict']['backbone.fc7.bias'])
            #self.fc_cls.weight = nn.Parameter(model['state_dict']['cls_head.fc_cls.weight'])
            #self.fc_cls.bias = nn.Parameter(model['state_dict']['cls_head.fc_cls.bias'])
            
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.005)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m, 1)
        
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.005)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

def generate_model(**kwargs):

    NUM_CLASSES = 101
    model = C3D(NUM_CLASSES)

    return model