import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=777, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
opts = parser.parse_args()

def get_feature(path):
    result = scipy.io.loadmat(path)
    query_feature = torch.FloatTensor(result['feature'])
    # query_feature = query_feature.cuda()
    query_feature = query_feature.view(1,-1)
    return query_feature


a = get_feature('../Market/pytorch/cam_feature/c1/0000426_0008_c1_f0000426_02.mat').view(-1,1)
print(a.shape)
b = get_feature('../Market/pytorch/query_feature/0008/0008_c5s1_000401_00.mat').view(1,-1)
print(b.shape)
print(torch.mm(b,a))
