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

import argparse
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
parser.add_argument('--dst_file_path', default=None, type=str)
args = parser.parse_args()

if args.dst_file_path is None:
    args.dst_file_path = args.file_path

x = torch.load(args.file_path)
state_dict = x['state_dict']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_k = '.'.join(k.split('.')[1:])
    new_state_dict[new_k] = v

x['state_dict'] = new_state_dict

torch.save(x, args.dst_file_path)