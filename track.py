# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
from selectors import EpollSelector
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, ft_net_dense, ft_net_swin, ft_net_NAS, PCB, PCB_test
from PIL import Image
import seaborn as sns
from scipy.signal import savgol_filter
import random
from threading import Thread, Event
from queue import Queue


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--method',default='static', type=str,help='static or straight correlation')
parser.add_argument('--reid_method',default='truth', type=str,help='ground truth or nn re-id')
parser.add_argument('--split',default='0', type=str,help='split of set')
parser.add_argument('--spatial',default=0, type=float,help='spatial correlation threhold')
parser.add_argument('--temporal',default=0, type=float,help='temporal correlation threhold')
parser.add_argument('--index',default=0, type=int,help='index')
opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.use_swin = config['use_swin']
opt.stride = config['stride']

method = opt.method
reid_method = opt.reid_method
cu_split = opt.split
spatial_t = opt.spatial
temporal_t = opt.temporal
index = opt.index
print("spatial correlation threhold: "+ str(spatial_t))
print("temporal correlation threhold: "+ str(temporal_t))
print("method: " + method)
print("re-id method: "+ reid_method)
print("cu_split: "+ str(cu_split))
print("index: "+ str(index))
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop)
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['train_all','gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['cameras','train_all','gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        #print(ff.data.cpu())
        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def save_feature(train_path):
    flag = 0
    for path, v in train_path:
        print(path)
        sub_index = []
        sub_index.append(flag)
        SingleSet = torch.utils.data.Subset(image_datasets['train_all'], sub_index)
        SingleLoader = torch.utils.data.DataLoader(SingleSet, batch_size=1, num_workers=0, shuffle=False)
        flag += 1
        seg = path.split('/')
        filename = seg[5].split('.')[0]+'.mat'
        path = '../Market/pytorch/train_feature/' + seg[4]
        if not os.path.exists(path):
            os.makedirs(path)
        des = os.path.join(path, filename)
        feat = extract_feature(model, SingleLoader)
        scipy.io.savemat(des, {'feature':feat.numpy()})

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
elif opt.use_swin:
    model_structure = ft_net_swin(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, stride = opt.stride)

if opt.PCB:
    model_structure = PCB(opt.nclasses)

#if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    #if opt.fp16:
    #    model = PCB_test(model[1])
    #else:
        model = PCB_test(model)
else:
    #if opt.fp16:
        #model[1].model.fc = nn.Sequential()
        #model[1].classifier = nn.Sequential()
    #else:
        model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
#dataloader_gallery = torch.utils.data.DataLoader(image_datasets['gallery'], batch_size=opt.batchsize, shuffle=False, num_workers=16)
#train_dataset = datasets.ImageFolder(os.path.join(data_dir,'train_all'), data_transforms)
train_path = image_datasets['train_all'].imgs




# Extract feature
#with torch.no_grad():
    #gallery_feature = extract_feature(model,dataloaders['gallery'])
    #query_feature = extract_feature(model,dataloaders['query'])
    #if opt.multi:
    #    mquery_feature = extract_feature(model,dataloaders['multi-query'])

# Save to Matlab for check
#result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
#scipy.io.savemat('pytorch_result.mat',result)

#print(opt.name)
#result = './model/%s/result.txt'%opt.name
#os.system('python evaluate_gpu.py | tee -a %s'%result)

# if opt.multi:
#     result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
#     scipy.io.savemat('multi_query.mat',result)


def get_frames(img_name):
    dict_cam_seq_max = {
        11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
        21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
        31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
        41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
        51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
        61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}

    frame = int(img_name.split('_')[2])
    camera = int(img_name.strip().split("_")[1][1])
    seq = int(img_name.strip().split("_")[1][3])
    re = 0
    for i in range(1, seq):
        re = re + dict_cam_seq_max[int(str(camera) + str(i))]
    re = re + frame
    return re


def get_id_plus(img_path):
    dict_cam_seq_max = {
        11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
        21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
        31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
        41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
        51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
        61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
    camera_id = []
    labels = []
    frames = []
    feature_paths = []
    for path, v in img_path:
        seg = path.split('/')
        route = '../Market/pytorch/train_feature/' + seg[4]
        mat = seg[5].split('.')[0]+'.mat'
        des = os.path.join(route, mat)
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = int(filename.strip().split("_")[1][1])
        seq = int(filename.strip().split("_")[1][3])
        # frame = filename[9:16]
        frame = int(filename.split('_')[2])
        re = 0
        for i in range(1, seq):
            re = re + dict_cam_seq_max[int(str(camera) + str(i))]
        re = re + frame
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
        frames.append(int(re))
        feature_paths.append(des)
    return camera_id, labels, frames, feature_paths


def get_feature(path):
    result = scipy.io.loadmat(path)
    query_feature = torch.FloatTensor(result['feature'])
    # query_feature = query_feature.cuda()
    query_feature = query_feature.view(1,-1)
    return query_feature

  


def spatial_temporal_distribution(camera_id, labels, frames, feature_paths, lower, upper):
    #class_num=751
    max_hist = 500
    spatial_temporal_sum = {}                       
    spatial_temporal_count = {}
    eps = 0.0000001
    interval = 200.0
    gallary = []
    have_labels = []
    for i in range(len(camera_id)):
        #print(feature_paths[i])
        label_k = labels[i]                 #### not in order, done
        cam_k = camera_id[i]-1              ##### ### ### ### ### ### ### ### ### ### ### ### # from 1, not 0
        frame_k = frames[i]
        feature_k = get_feature(feature_paths[i])
        #print(label_k, cam_k, frame_k)
        if  frame_k > upper or frame_k < lower:
            continue
        if len(gallary) == 0:
            # first feature
            gallary.append(feature_k)
            id = 0
            spatial_temporal_sum[(id, cam_k)] = frame_k
            spatial_temporal_count[(id, cam_k)] = 1
            have_labels.append(label_k)
        else:
            tmp = feature_k.cuda()
            tmp = tmp.view(-1, 1)
            check = False
            id = 0
            for feat in gallary:
                feat = feat.cuda()
                similarity = torch.mm(feat, tmp)
                if similarity > 0.6:
                    check = True
                    # print("Paired!!")
                    # print(have_labels[id], label_k)
                    break
                id += 1
            if not check:
                # if new person
                if label_k in have_labels:
                    print("Miss!!")
                gallary.append(feature_k)
                spatial_temporal_sum[(id, cam_k)] = frame_k
                spatial_temporal_count[(id, cam_k)] = 1
                have_labels.append(label_k)
            else:
                # if exist
                if (id, cam_k) in spatial_temporal_sum.keys():
                    # check whether shown up in current cam
                    tmp_f = spatial_temporal_sum[(id, cam_k)]
                    spatial_temporal_sum[(id, cam_k)] = frame_k + tmp_f
                    tmp_c = spatial_temporal_count[(id, cam_k)]
                    spatial_temporal_count[(id, cam_k)] = tmp_c + 1
                else:
                    # if new to certain cam
                    spatial_temporal_sum[(id, cam_k)] = frame_k
                    spatial_temporal_count[(id, cam_k)] = 1
    
    
    ids = []
    for key in spatial_temporal_count.keys():
        if key[0] not in ids:
            ids.append(key[0])
    class_num = len(ids)
    spatial_temporal_avg = np.zeros((class_num,8))  
    for key in spatial_temporal_count.keys():
        spatial_temporal_avg[key[0]][key[1]] = spatial_temporal_sum[key]/(spatial_temporal_count[key] + eps)

   
    
    spatial_distribution = np.zeros((6,7))
    cam_sum = np.zeros(6)
    temporal_distribution = np.zeros((6,6,max_hist))
    res = np.zeros((4,7))
    for i in range(class_num):
        tmp = {}
        for j in range(6):
            tmp[j] = spatial_temporal_avg[i][j]
        list1 = sorted(tmp.items(), key = lambda x:x[1],reverse=True)
        for j in range(6):
            if list1[j][1] == 0:
                break
            cam_sum[list1[j][0]] += 1

            if j == 5:
                break
            if list1[j+1][1] != 0:
                des = list1[j][0]
                start = list1[j+1][0]
                spatial_distribution[start][des] += 1
                # if start == 1:
                #     print(cam_sum[start])
                #     s = 0
                #     for k in range(6):
                #         s += spatial_distribution[start][k]
                #     print(s)
                if des == 2:
                    res[int(list1[j][1]/110000)][start] += 1

                diff = list1[j][1] - list1[j+1][1]
                hist_ = int(diff/interval)

                if hist_ > 500:
                    print("person:" + str(i) + " start:" + str(start) + " at:" + str(list1[j+1][1]) + " des:" + str(des) + " at:" + str(list1[j][1]))
                    continue
            
                temporal_distribution[start][des][hist_] += 1
    # print(cam_sum[1])
    # s = 0
    # for k in range(6):
    #     s += spatial_distribution[1][k]
    # print(s)
    print(res)
    np.save('in_2.txt', res)
    # sum_ = np.sum(temporal_distribution,axis=2)
    # for i in range(6):
    #     print("cam" + str(i) + ": "+str(cam_sum[i]))
    #     numExit = cam_sum[i]
    #     for j in range(6):
            
    #         # if i == 1:
    #         #     print(spatial_distribution[i][j])
    #         numExit -= spatial_distribution[i][j]
    #         spatial_distribution[i][j] /= cam_sum[i]
    #     spatial_distribution[i][6] = numExit / cam_sum[i]
    #     for k in range(6):
    #         if sum_[i][k] == 0:
    #             temporal_distribution[i][k][:] = 0
    #         else:
    #             temporal_distribution[i][k][:] /= sum_[i][k]

    # code for drawing heatmap of spatial correlation
            
    # x_label = ["cam1","cam2","cam3","cam4","cam5","cam6","Exit"]
    # y_label = ["cam1","cam2","cam3","cam4","cam5","cam6"]
    # ax = sns.heatmap(spatial_distribution, linewidth=0.5, annot=True, cmap="YlGnBu", xticklabels=x_label, yticklabels=y_label)
    # ax.set_title('Spatial Correlation')
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # # plt.show()
    # s = ax.get_figure()
    #s.savefig('./correlation_all.jpg', dpi = 300, bbox_inches = 'tight')


    
    return spatial_distribution, temporal_distribution                    # [to][from], to xxx camera, from xxx camera



def bound4temporal_cor(arr, t_threshold):
    total = np.sum(arr)
    lower = 0
    upper = 499
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
        if (i == 0) and (sum > total * t_threshold/2):
            break
        if(sum > total * t_threshold/2):
            lower = i
            break
    sum = 0
    for i in range(len(arr)):
        sum += arr[499-i]
        if(sum > total * t_threshold/2):
            upper = 499-i
            break
    return lower, upper

def correlation_filter(f_curr, f_q, c_q, spatial_correlation, temporal_correlation,data, s_threshold, t_threshold,flag):
    queryList = []
    for i in range(1,7):
        if i == int(c_q):
            continue
        # dataPath = "../Market/pytorch/cameras/c" + str(i)
        if flag == True:
            if spatial_correlation[int(c_q)-1][i-1] < s_threshold:
                continue
        else:
            if spatial_correlation[int(c_q)-1][i-1] < s_threshold/10:
                continue
        img_names = data[i-1]
        tmp_lower, tmp_upper = bound4temporal_cor(temporal_correlation[int(c_q)-1][i-1], t_threshold)
        tmp_lower1, tmp_upper1 = bound4temporal_cor(temporal_correlation[int(c_q)-1][i-1], t_threshold/10)
        frame_lower = tmp_lower * 200 + f_q
        frame_upper = tmp_upper * 200 + f_q
        frame_lower1 = tmp_lower1 * 200 + f_q
        frame_upper1 = tmp_upper1 * 200 + f_q
        # print("lower: " + str(frame_lower) + " upper: " + str(frame_upper))
        # print("lower1: " + str(frame_lower1) + " upper1: " + str(frame_upper1))
        # if i == 2:
        #     
        #if i == 2:
            #print(frame_lower)
        # frame_lower = f_curr-50
        # frame_upper = f_curr + 50
        #print("f_q: " + str(f_q) +" lower: " + str(frame_lower) + " upper: " + str(frame_upper))
        for image in img_names:
            fImage = int(image[0:7])
            #print(fImage)
            if fImage <= f_q:
                continue
            if (flag == True) :
                if fImage < frame_lower or fImage < f_curr-50:
                    continue
                if fImage > f_curr+50 or fImage > frame_upper:
                    break
            else:
                if spatial_correlation[int(c_q)-1][i-1] > s_threshold:
                    if fImage < f_curr-50:
                        continue
                    if fImage > f_curr+50 or fImage > frame_upper1:
                        break

                    if not((fImage>frame_lower1 and fImage < frame_lower) or (fImage > frame_upper and fImage < frame_upper1)):
                        continue
                else:
                    if fImage < frame_lower1 or fImage < f_curr-50:
                        continue
                    if fImage > f_curr+50 or fImage > frame_upper1:
                        break

            if (abs(f_curr - fImage) <= 50) :
                if image not in queryList:
                    queryList.append(image)
    return queryList



def priorityTracking(p_q, f_q, c_q, feat_q, spatial_correlation, temporal_correlation, method, s_threshold, t_threshold,flag):
    exit_f = 450000
    f_curr = f_q
    queryNum = 0
    matched = [] 
    # correlationDict = {}
    # count = 0
    feat_q = feat_q.cuda()
    feat_q = feat_q.view(1, -1)
    data = []
    # print(spatial_correlation)
    for i in range(1,7):

        dataPath = "../Market/pytorch/cameras/c" + str(i)
        img_names = sorted(os.listdir(dataPath))
        data.append(img_names)
    # print(spatial_correlation[int(c_q)-1])
    while((f_curr - f_q) <= exit_f):
        # count = int((f_curr-f_q)/200)
        # for i in range(1,7):
        #     if i == int(c_q):
        #         continue    
        #     correlationDict[i] = spatial_correlation[i-1][int(c_q)-1] * temporal_correlation[i-1][int(c_q)-1][count]
        # sortedDict = sorted(correlationDict.items(), key=lambda item:item[1], reverse=True)
        # for i in range(len(sortedDict)):
            # targetCam = sortedDict[i][0]
        
        qList = correlation_filter(f_curr, f_q, c_q, spatial_correlation, temporal_correlation,data, s_threshold, t_threshold,flag)
        #print(qList)

        # tmp_match = qList[0]
        tmp_match = None
        tmp_max = 0
        for obj in sorted(qList):
            #print(obj)
            queryNum += 1
            if method == "truth":
                person = obj.split('_')[1]
                if person == p_q:
                    matched.append(obj)
                    break
            else:
                cam = obj.split('_')[2]
                name = obj.split('.')[0] + '.mat'
                tmp_path = '../Market/pytorch/cam_feature/' + cam +'/' + name
                tmp_feat = get_feature(tmp_path)
                tmp_feat = tmp_feat.cuda()
                tmp_feat = tmp_feat.view(-1,1)
                if torch.mm(feat_q, tmp_feat) > tmp_max:
                    tmp_max = torch.mm(feat_q, tmp_feat)
                    tmp_match = obj

                    
        if tmp_max > 0.62:
            matched.append(tmp_match)
            break


        f_curr += 101
    # if queryNum == 0:
    #     print(spatial_correlation[int(c_q)-1])
    return matched, queryNum


def simpleFilter(camNum, frame,data):
    queryList = []
    for i in range(1,7):
        if i == int(camNum):
            continue
        img_names = data[i-1]
        for image in img_names:
            fImage = int(image[0:7])
            if fImage <= frame:
                continue
            if fImage > frame + 100:
                break
            else:
                queryList.append(image)
                #print(image)
    return queryList




def commonTracking(p_q, f_q, c_q, feat_q, method):
    f_curr = f_q
    exit_f = 400000
    queryNum = 0
    matched = []
    feat_q = feat_q.cuda()
    feat_q = feat_q.view(1, -1)
    data = []
    for i in range(1,7):

        dataPath = "../Market/pytorch/cameras/c" + str(i)
        img_names = sorted(os.listdir(dataPath))
        data.append(img_names)
    while((f_curr - f_q) <= exit_f):
        qList = simpleFilter(c_q, f_curr,data)
        #print(qList)
        #print(f_curr, ":", len(qList))
        tmp_match = None
        tmp_max = 0
        for obj in sorted(qList):
            queryNum += 1
            person = obj.split('_')[1]
            if method == "truth":
                if person == p_q:
                    matched.append(obj)
                    break
            else:
                cam = obj.split('_')[2]
                name = obj.split('.')[0] + '.mat'
                tmp_path = '../Market/pytorch/cam_feature/' + cam +'/' + name
                tmp_feat = get_feature(tmp_path)
                tmp_feat = tmp_feat.cuda()
                tmp_feat = tmp_feat.view(-1,1)
                if torch.mm(feat_q, tmp_feat) > tmp_max:
                    tmp_max = torch.mm(feat_q, tmp_feat)
                    tmp_match = obj

                    
        if tmp_max > 0.62:
            matched.append(tmp_match)
            break
        f_curr = f_curr + 100
    return matched, queryNum



def tracking(split, s_threshold, t_threshold, method,index):
    # transform_train_list = [
    #         transforms.Resize(144, interpolation=3),
    #         transforms.RandomCrop((256,128)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['train_all']}
    train_path = image_datasets['train_all'].imgs
    train_cam, train_label, train_frames, train_feature_paths = get_id_plus(train_path)
    # for tmp_sp in split :
    #     for i in range(tmp_sp):

    #         model_path = './correlation_split' + str(tmp_sp) + '_' + str(i) + '.mat'
    #         print(model_path)
    #         set_len = int(440000 / tmp_sp)
    #         lower = set_len * i
    #         upper = set_len * (i+1)
    #         print(lower, upper)
    #         spatial_distribution, temporal_distribution  = spatial_temporal_distribution(train_cam, train_label, train_frames, train_feature_paths, lower, upper)
            # scipy.io.savemat(model_path, {'spatial':spatial_distribution, 'temporal': temporal_distribution})
    # spatial_distribution, temporal_distribution  = spatial_temporal_distribution(train_cam, train_label, train_frames, train_feature_paths, 0, 440000)
    # #scipy.io.savemat(model_path, {'spatial':spatial_distribution, 'temporal': temporal_distribution})
    
    # if method == 'dynamic':
    #     #spatial_distribution_2, temporal_distribution_2  = spatial_temporal_distribution(train_cam, train_label, train_frames, train_feature_paths, 2)
    #     model_path = './correlation_2.mat'
    #     result = scipy.io.loadmat(model_path)
    #     spatial_distribution_2 = result['spatial']
    #     temporal_distribution_2 = result['temporal']
    #     #scipy.io.savemat(model_path, {'spatial':spatial_distribution_2, 'temporal': temporal_distribution_2})
    #     #spatial_distribution_3, temporal_distribution_3  = spatial_temporal_distribution(train_cam, train_label, train_frames, train_feature_paths, 3)
    #     model_path = './correlation_3.mat'
    #     result = scipy.io.loadmat(model_path)
    #     spatial_distribution_3 = result['spatial']
    #     temporal_distribution_3 = result['temporal']
    #     #scipy.io.savemat(model_path, {'spatial':spatial_distribution_3, 'temporal': temporal_distribution_3})
    #     #spatial_distribution_4, temporal_distribution_4  = spatial_temporal_distribution(train_cam, train_label, train_frames, train_feature_paths, 4)
    #     model_path = './correlation_4.mat'
    #     result = scipy.io.loadmat(model_path)
    #     spatial_distribution_4 = result['spatial']
    #     temporal_distribution_4 = result['temporal']
    #     #scipy.io.savemat(model_path, {'spatial':spatial_distribution_4, 'temporal': temporal_distribution_4})
    totals = []
    misses = []
    
    for tmp_sp in split:
        test_itr = 750
        dataPath = "../Market/pytorch/query/"
        queryObj = []
        total = 0
        miss_included = 0
        total_static = 0
        total_query = 0
        static_miss =0
        predict_miss =0
        common_err = 0
        common_miss = 0
        dynamic_miss =0
        static_err = 0
        dynamic_err = 0
        predict_err = 0
        static = 0
        dynamic = 0
        total_dynamic = 0
        predict = 0
        total_predict = 0
        total_predict_less = 0
        common_retrieved = 0
        predict_retrieved = 0
        common_retrieved_less = 0
        predict_retrieved_less = 0
        predict_correct = 0
        predict_correct_less = 0
        common_correct = 0
        common_correct_less = 0
        commonTotal_static_less = 0
        
        cdf = np.zeros(50)

        # commonTotal = 0
        commonTotal_static = 0
        
        objList = os.listdir(dataPath)
        for obj in objList:
            if not obj[0] == '.':
                queryObj.append(obj)
        queryObj = sorted(queryObj)
        # print(len(queryObj))
        miss = []
        # miss_static = []
        wrong_static = []
        # rates_static = []
        # wrong = []
        rates = []
        # e = []
        e_static = []
        fo = open("dynamic_stepsize.txt", "a")
        for i in range(test_itr):

            queryFrame = os.listdir(dataPath + queryObj[i])
            
            # index = random.randint(0,len(queryFrame)-1)

            
            tmpQuery = queryFrame[index]
            #tmpQuery = '0006_c6s4_002202_00.jpg'
            tmpQuery = tmpQuery.split('.')[0] + '.jpg'
            #tmpQuery = '0006_c3s3_075694_00.jpg'
            print(tmpQuery)
            tmpFrame = get_frames(tmpQuery)
            # print(tmpFrame)
            
            feat_path = os.path.join('../Market/pytorch/query_feature', queryObj[i], tmpQuery[:-4]+'.mat')
            #feat_path = os.path.join('../Market/pytorch/query_feature/0006', tmpQuery[:-4]+'.mat')
            feat_q = get_feature(feat_path)
            tmpObj = tmpQuery.split('_')[0]
            tmpCamera = tmpQuery.split('_')[1][1]
            # model_path = './correlation_predict_'+str(tmp_sp)+'.mat'
            # result = scipy.io.loadmat(model_path)
            # spatial_distribution = result['spatial']
            model_path = './correlation_split8_0.mat'
            result = scipy.io.loadmat(model_path)
            temporal_distribution = result['temporal']
            
            tmp_split = int(tmp_sp)
            seq = int(tmpFrame*tmp_sp/440000)
            # tmp_spatial = spatial_distribution[int(seq*tmp_split)]
            # tmp = spatial_distribution[0]
            # for k in range(int(seq*tmp_split+1), int(seq*tmp_split+32/tmp_split + 1)):
            #     tmp_spatial = [[tmp[i][j] + spatial_distribution[k][i][j]  for j in range(len(tmp[0]))] for i in range(len(tmp))]
            #     tmp = tmp_spatial.copy()
            # tmp_spatial = [[tmp[i][j]/tmp_split  for j in range(len(tmp[0]))] for i in range(len(tmp))]



            model_path = './correlation_split14_0.mat'
            result = scipy.io.loadmat(model_path)
            
            temporal_distribution_start = result['temporal']
            spatial_distribution_start = result['spatial']
            # temporal_distribution_start = np.array(temporal_distribution_start)
            # print(temporal_distribution_start.shape)
            # matched_dynamic, queryNum_dynamic = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , tmp_spatial, temporal_distribution, reid_method)
            model_path = './correlation_split14_13.mat'
            result = scipy.io.loadmat(model_path)
            spatial_distribution_end = result['spatial']
            temporal_distribution_end = result['temporal']
            for k in range(len(temporal_distribution_start)):
                temporal_distribution[k] = [[temporal_distribution_start[k][i][j]/2+ temporal_distribution_end[k][i][j]/2  for j in range(len(temporal_distribution_start[0][0]))] for i in range(len(temporal_distribution_start[0]))]
            
            # temporal_distribution = np.array(temporal_distribution)
            # print(temporal_distribution.shape)
            if method == "static":
                spatial_distribution = [[spatial_distribution_start[i][j]/2+ spatial_distribution_end[i][j]/2  for j in range(len(spatial_distribution_start[0]))] for i in range(len(spatial_distribution_start))]
                matched_predicted, queryNum_predicted = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method, s_threshold, t_threshold, True)
            elif method == "dynamic":
                model_path = './correlation_split' + str(tmp_sp) + '_' + str(seq) + '.mat'
                result = scipy.io.loadmat(model_path)
                spatial_distribution = result['spatial']
                # temporal_distribution = result['temporal']
                matched_predicted, queryNum_predicted = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method, s_threshold, t_threshold, True)
            else:
                tmp_spatial_0 = [[spatial_distribution_end[i][j] - spatial_distribution_start[i][j]  for j in range(len(spatial_distribution_start[0]))] for i in range(len(spatial_distribution_start))]
                
                tmp_spatial_1 = [[tmp_spatial_0[i][j]/int(tmp_sp)  for j in range(len(tmp_spatial_0[0]))] for i in range(len(tmp_spatial_0))]
                spatial_distribution = [[spatial_distribution_start[i][j]+ tmp_spatial_1[i][j]*int(seq)  for j in range(len(tmp_spatial_1[0]))] for i in range(len(tmp_spatial_1))]
                # print(tmp_spatial_2[0])
                matched_predicted, queryNum_predicted = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method, s_threshold, t_threshold, True)
                # if method == 'dynamic':
            #     if tmpFrame < 100000:
            #         matched, queryNum = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method)
            #     elif tmpFrame < 200000:
            #         matched, queryNum = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution_2, temporal_distribution_2, reid_method)
            #     elif tmpFrame < 300000:
            #         matched, queryNum = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution_3, temporal_distribution_3, reid_method)
            #     else:
            #         matched, queryNum = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution_4, temporal_distribution_4, reid_method)
            # check = False
            # check_static = False
            # if len(matched_static) == 0:
            #     # print(tmpQuery)
            #     miss.append(tmpQuery)
            # else:
            #     # total += queryNum_static
            #     check = True
            # if len(matched_static) == 0:
            #     print(tmpQuery)
            #     miss_static.append(tmpQuery)
            #     check_static = True

            
            
            # print(queryNum)
            # print(matched)
            # print(matched_static)
            # print(queryNum_static)
            
            commonMatched, commonQueryNum = commonTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q, reid_method)
            
            try:
                predict_obj = matched_predicted[0].split('_')[1]
            except:
                predict_obj = ''
            try:
                common_obj = commonMatched[0].split('_')[1]
            except:
                common_obj = ''
            # print("tmpObj is " + tmpObj)
            # print("common_obj is " + common_obj)
            # if global search find the correct one
            commonTotal_static += commonQueryNum
            total_predict += queryNum_predicted
            # common_flag = False
            # predict_flag = False
            predict_err_flag = False
            predict_correct_flag = False
            common_err_flag = False
            common_correct_flag = False
            if common_obj == tmpObj:

                common_correct+=1
                common_correct_flag = True
                # if our method find the correct one
                if matched_predicted == commonMatched:
                    
                    predict_correct +=1
                    predict_correct_flag = True
                # if our method does not find the correct answer    
                else:
                    # find no one
                    if len(matched_predicted) == 0:
                        matched_predicted, queryNum_predicted_tmp = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method, s_threshold, t_threshold, False)
                        total_predict += queryNum_predicted_tmp
                        queryNum_predicted += queryNum_predicted_tmp
                        if matched_predicted == commonMatched:
                            predict_correct_flag = True
                            predict_correct +=1
                        else:
                            if len(matched_predicted) != 0:
                                predict_err +=1
                                predict_err_flag = True


                    # find the different one
                    else:
                        if(predict_obj == tmpObj) and ( abs(int(matched_predicted[0].split('_')[0])-int(commonMatched[0].split('_')[0])) < 100):
                            predict_correct+=1
                            predict_correct_flag = True
                        else:
                            predict_err += 1
                            predict_err_flag = True

            # if global search find the wrong one
            else:
                
                # if global search find no one, our method can not find one as well
                if len(commonMatched) == 0:
                    total_predict += (commonQueryNum-queryNum_predicted)

                    predict_miss += 1
                    common_miss += 1

                # if global search find the wrong one
                else:
                    common_err += 1
                    common_err_flag = True
                    # if our method find no one
                    if len(matched_predicted) == 0:
                        matched_predicted, queryNum_predicted_tmp = priorityTracking(tmpObj, int(tmpFrame), tmpCamera, feat_q , spatial_distribution, temporal_distribution, reid_method, s_threshold, t_threshold, False)
                        # total_predict += queryNum_predicted
                        total_predict += queryNum_predicted_tmp
                        queryNum_predicted += queryNum_predicted_tmp
                        try:
                            predict_obj = matched_predicted[0].split('_')[1]
                        except:
                            predict_obj = ''
                        if predict_obj == tmpObj:
                            predict_correct_flag = True
                            predict_correct +=1
                        else:
                            if len(matched_predicted) != 0:
                                predict_err +=1
                                predict_err_flag = True


                    # if our method find one
                    else:
                        # if we find the correct one
                        if predict_obj == tmpObj:
                            predict_correct_flag = True
                            predict_correct += 1
                        # if we find the wrong one
                        else:
                            predict_err_flag = True
                            predict_err += 1
            seq = commonQueryNum/50
            if(seq > 49):
                seq = 49
            cdf[int(seq)] += 1
            if (commonQueryNum > 100):
                if predict_correct_flag or predict_err_flag:
                    predict_retrieved_less += 1
                    if predict_correct_flag:
                        predict_correct_less += 1
                if common_correct_flag or common_err_flag:
                    common_retrieved_less  += 1
                    if common_correct_flag:
                        common_correct_less+=1
                commonTotal_static_less += commonQueryNum
                total_predict_less += queryNum_predicted
                total_query += 1
                 
            print(matched_predicted)
            print(queryNum_predicted)
            print(commonMatched)
            print(commonQueryNum)
            common_retrieved = common_correct + common_err
            predict_retrieved = predict_correct + predict_err
            print(" predict_correct: " + str(predict_correct) + " common_correct: " + str(common_correct) + " predict_retrieved: " + str(predict_retrieved)+ " common_retrieved: " + str(common_retrieved)+ " total query: " + str(total_query) + '\n')




            
            
            # if matched_static != commonMatched:
            #     wrong_static.append(tmpQuery)
            #     check = False
            # if check or check_static:
            #     total+= queryNum_static
            #     commonTotal_static += commonQueryNum
            #     miss_included = total
            # if matched_static != commonMatched and len(matched_static) == 0 :
            #     miss_included += commonQueryNum
            # if (i%50 == 0) & (i != 0):
            #     rate = float(commonTotal_static)/float(total_static)
            #     rates_static.append(rate)
            #     print("rate:" + str(float(commonTotal_static)/float(total_static)))
            # if matched != commonMatched:
            #     if len(matched) == len(commonMatched) == 0:
            #         e.append(tmpQuery)
            #     else:
            #         wrong.append(tmpQuery)
            #     check = True
            # if not check:
            #     total += queryNumxw
            #     commonTotal += commonQueryNum
            # if (i%50 == 0) & (i != 0):
            #     rate = float(commonTotal)/float(total)
            #     rates.append(rate)
            #     print("rate:" + str(float(commonTotal)/float(total)))
            # # break
        # misses.append(len(miss)-len(e_static))
        # rates.append(float(commonTotal_static)/float(total))
        # print("dynamic method_split" + str(tmp_sp) + ": missNum: " +str(dynamic_miss)+ ": errNum: " +str(dynamic_err) + "ratio: " + str(float(commonTotal_static)/float(dynamic))  + "ratio(miss included): " + str(float(commonTotal_static)/float(total_dynamic))+ "\n")
        # fo.write("dynamic method_split" + str(tmp_sp) + ": missNum: " +str(dynamic_miss)+ ": errNum: " +str(dynamic_err) + "ratio: " + str(float(commonTotal_static)/float(dynamic))  + "ratio(miss included): " + str(float(commonTotal_static)/float(total_dynamic))+ "\n")
        # print("static method_split" + str(tmp_sp) + ": missNum: " +str(static_miss)+ ": errNum: " +str(static_err) + "ratio: " + str(float(commonTotal_static)/float(static))  + "ratio(miss included): " + str(float(commonTotal_static)/float(total_static))+ "\n")
        # fo.write("static method_split" + str(tmp_sp) + ": missNum: " +str(static_miss)+ ": errNum: " +str(static_err) + "ratio: " + str(float(commonTotal_static)/float(static))  + "ratio(miss included): " + str(float(commonTotal_static)/float(total_static))+ "\n")
        print("predict_correct: "+ str(predict_correct_less) + " total query: " + str(total_query) + "\n")
        common_retrieved = common_correct + common_err
        predict_retrieved = predict_correct + predict_err
        filename = 'global' + str(tmp_sp)
        print(cdf)
        np.savetxt(filename,cdf)
        fo.write("index = "+str(index)+"\n")
        if method == "static":
            print("static method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            fo.write("static method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            print("less static method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
            fo.write("less static method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
        elif method == "dynamic":
            print("dynamic method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            fo.write("dynamic method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            print("less dynamic method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
            fo.write("less dynamic method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
    
        else:
            print("baseline method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            fo.write("baseline method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct/750))+ " precision: " +str(float(predict_correct/predict_retrieved)) + " searched frames: " + str(total_predict) + "\n")
            print("less baseline method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
            fo.write("less baseline method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(predict_correct_less/total_query))+ " precision: " +str(float(predict_correct_less/predict_retrieved_less)) + " searched frames: " + str(total_predict_less) + "\n")
    
        
        print("common method:"  + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(common_correct/750))+ " precision: " +str(float(common_correct/common_retrieved)) + " searched frames: " + str(commonTotal_static) + "\n")
        fo.write("common method:" + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(common_correct/750))+ " precision: " +str(float(common_correct/common_retrieved)) + " searched frames: " + str(commonTotal_static) + "\n")
        print("less common method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(common_correct_less/total_query))+ " precision: " +str(float(common_correct_less/common_retrieved_less)) + " searched frames: " + str(commonTotal_static_less) + "\n")
        fo.write("less common method_split" + str(tmp_sp) + " spatial threshold: " +str(s_threshold)+ " temporal threshold: " +str(t_threshold)+ " recall: " +str(float(common_correct_less/total_query))+ " precision: " +str(float(common_correct_less/common_retrieved_less)) + " searched frames: " + str(commonTotal_static_less) + "\n")
    
        fo.flush()
        fo.close()
    # print(rates)
    # print(misses)
        # print("miss:")
        # print(len(miss))
        # print("wrong:")
        # print(len(wrong))
        # print(rates)

        # print("static method: "+ str(total_static) + "common method " + str(commonTotal_static) + "ratio: " + str(float(commonTotal_static)/float(total_static)))
        # print("miss:")
        # print(len(miss_static)-len(e_static))
        # print("wrong:")
        # print(len(wrong_static))
        # print(rates_static)


# def edge_find(que, time_lower, time_upper, feat_q, cam):

#     path = '../Market/pytorch/cam_feature/c' + str(cam)
#     img_names = sorted(os.listdir(path))
#     bound = len(img_names)
#     i = 0
#     queryNum = 0
#     matched = []
#     first = True
#     while True:
#         if i > bound-1:
#             break
#         flag, evt = que.get()
#         if flag > 0:
#             evt.set()
#             break
#         while first:
#             if int(img_names[i].split('_')[0]) < time_lower :
#                 i += 1
#             else:
#                 first = False
#         tmp_person = img_names[i]
#         if int(img_names[i].split('_')[0]) > time_upper:
#             evt.set()
#             break
#         queryNum += 1
#         if tmp_person.split('_')[1] == feat_q:
#             evt.set()
#             matched.append(tmp_person)
#             break
#         i += 1

#     return matched, queryNum




def main():
    # split = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    i = int(cu_split)
    s_threshold = float(spatial_t)
    t_threshold = float(temporal_t)
    # if index == 0:
    #     in_split = [32,36]
    # if index == 1:
    #     in_split = [40,44]
    # if index == 2:
    #     in_split = [48,52]
    # if index == 3:
    #     in_split = [56,60]
    in_split = [i]
    tracking(in_split, s_threshold, t_threshold, method,index)
    # train_path = image_datasets['cameras'].imgs
    # flag = 0
    # for path, v in train_path:
    #     print(path)
    #     sub_index = []
    #     sub_index.append(flag)
    #     SingleSet = torch.utils.data.Subset(image_datasets['cameras'], sub_index)
    #     SingleLoader = torch.utils.data.DataLoader(SingleSet, batch_size=1, num_workers=0, shuffle=False)
    #     flag += 1
    #     seg = path.split('/')
    #     filename = seg[5].split('.')[0]+'.mat'
    #     path = '../Market/pytorch/cam_feature/' + seg[4]
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     des = os.path.join(path, filename)
    #     feat = extract_feature(model, SingleLoader)
    #     scipy.io.savemat(des, {'feature':feat.numpy()})


if __name__ == '__main__':
    main()


# def parallel_correlation_tracking(p_q, f_q, c_q, feat_q, spatial_correlation, temporal_correlation):
#     exit_f = 450000
#     f_curr = f_q
#     queryNum = 0
#     matched = [] 

#     feat_q = feat_q.cuda()
#     feat_q = feat_q.view(1, -1)
#     search_list = []
#     searched = {}
#     for i in range(1,7):
#         if spatial_correlation[i-1][int(c_q)-1] < 0.05:
#             continue
#         else:
#             tmp_lower, tmp_upper = bound4temporal_cor(temporal_correlation[i-1][int(c_q)-1])
#             search_list.append((i,tmp_lower, tmp_upper))
#             searched[i] = 0
#     threads = []
#     que = Queue()
#     evt = Event()
#     ddl = 0
#     que.put((ddl,evt))
#     while(True):
#         # trigger searching
#         # print("looping...")
#         for i in range(len(search_list)):
#             time_lower = search_list[i][1] * 200 + f_q
#             time_upper = search_list[i][2] * 200 + f_q
#             tmp_cam = search_list[i][0]
#             # print(time_lower, time_upper, tmp_cam)
#             if time_lower < f_curr and time_upper > f_curr and searched[tmp_cam] == 0:
#                 print("trigger searching in cam" + str(tmp_cam))
#                 t = MyThread(func=edge_find, args=(que, time_lower, time_upper, p_q, tmp_cam))
#                 threads.append(t)
#                 t.start()
#                 searched[search_list[i][0]] = 1

#         # check any edge find the target? if so, stop others
#         found = False
#         for t in threads:
#             if not t.is_alive():
#                 tmp_matched, tmp_searchNum = t.get_result()
#                 queryNum += tmp_searchNum
#                 threads.remove(t)
            
#                 if len(tmp_matched) != 0:
#                     found = True
#                     matched.append(tmp_matched)
#                     ddl = int(tmp_matched[0].split('_')[0])
#                     break


#         if found:
#             que.put((ddl,evt))
#             evt.wait()

#             for t in threads:

#                 tmp_matched, tmp_searchNum = t.get_result()
#                 queryNum += tmp_searchNum


#         # qList = correlation_filter(f_curr, f_q, c_q, spatial_correlation, temporal_correlation)
#         # #print(qList)
        
#         # for obj in sorted(qList):
#         #     #print(obj)
#         #     queryNum += 1
#         #     person = obj.split('_')[1]
#         #     if person == p_q:
#         #         matched.append(obj)
#         #         break
#         #     # cam = obj.split('_')[2]
#         #     # name = obj.split('.')[0] + '.mat'
#         #     # tmp_path = '../Market/pytorch/cam_feature/' + cam +'/' + name
#         #     # tmp_feat = get_feature(tmp_path)
#         #     # tmp_feat = tmp_feat.cuda()
#         #     # tmp_feat = tmp_feat.view(-1,1)
#         #     # if torch.mm(feat_q, tmp_feat) > 0.6:
#         #     #     matched.append(obj)
#         #     #     break
#         if len(matched) != 0:
#             break

#         f_curr += 101
#         if((f_curr - f_q) > exit_f):
#             print("time exceed!")
#             break
#     return matched, queryNum

