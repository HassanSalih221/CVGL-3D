# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import ttach as tta

import time

from cvcities_base.model import TimmModel
from cvcities_base.dataset.university import get_transforms
from utils.image_folder import CustomData160k_sat, CustomData160k_drone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='4', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/home/zhanghy/MM/data/University-1652/test',type=str, help='./test_data')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--query_name', default='query_street_name.txt', type=str,help='load query image')
opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
test_dir = opt.test_dir
query_name = opt.query_name
ms = [1]

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = (448, 448)

val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std)

data_dir = test_dir

image_datasets = {}
image_datasets['gallery_satellite'] = CustomData160k_sat(os.path.join(data_dir, 'workshop_gallery_satellite'), val_transforms)
image_datasets['query_street'] = CustomData160k_drone(os.path.join(data_dir,'workshop_query_street'), val_transforms, query_name = query_name)
print(image_datasets.keys())


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                shuffle=False, num_workers=16) for x in
                ['gallery_satellite','query_street']}

use_gpu = torch.cuda.is_available()

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

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1


tta_transforms = tta.Compose([
    tta.HorizontalFlip(),
    tta.Rotate90(angles=[0, 90]),
])

def extract_feature(model, dataloader):
    features = []
    model_tta = tta.ClassificationTTAWrapper(model, tta_transforms)  # 使用 TTA 包装模型
    model_tta.eval()
    
    if use_gpu:
        model_tta = model_tta.to(device)

    for data in dataloader:
        img, _ = data
        input_img = Variable(img.to(device))

        with torch.no_grad():
            outputs = model_tta(input_img)  # 自动应用所有 TTA 变换并融合结果

        feature = F.normalize(outputs, dim=-1)
        features.append(feature.cpu().data)

    features = torch.cat(features, dim=0)
    return features

def get_SatId_160k(img_path):
    labels = []
    paths = []
    for path,v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def compute_distmat(query, gallery):
    # L2 normalize
    query = query / query.norm(dim=1, keepdim=True)
    gallery = gallery / gallery.norm(dim=1, keepdim=True)
    distmat = euclidean_distances(query.cpu().numpy(), gallery.cpu().numpy())
    return distmat

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # Reference: https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/re_ranking.py
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
        np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
         axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        weight = np.exp(-original_dist[i, k_reciprocal_index])
        V[i, k_reciprocal_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, query_num:]
    V_qe = V[:query_num, :]
    V_ge = V[query_num:, :]
    dist = 1 - np.dot(V_qe, V_ge.T)
    final_dist = (1 - lambda_value) * original_dist + lambda_value * dist
    return final_dist

def get_result_rank10(qf,gf,gl):
    query = qf.view(-1,1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10


if __name__ == "__main__":
    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    
    class Configuration:
        
        backbone_arch = 'dinov2_vitl14'
        model_name = 'dinov2_vitl14_MixVPR'
        agg_arch = 'MixVPR'
        agg_config = {'in_channels': 1024,
                        'in_h': 32,  # 受输入图像尺寸的影响
                        'in_w': 32,
                        'out_channels': 1024,
                        'mix_depth': 2,
                        'mlp_ratio': 1,
                        'out_rows': 4}
        layer1 = 7
        checkpoint_start = ''
        # point clip
        num_views: int = 10
        backbone_name: str = 'RN101'
        backbone_channel: int = 512
        adapter_ratio: float = 0.6
        adapter_init: float = 0.5
        adapter_dropout: float = 0.1
        use_pretrained: bool = True
        
    args = Configuration()
    model = TimmModel(model_name=args.model_name, args=args,
                        pretrained=True,
                        img_size=img_size, backbone_arch=args.backbone_arch, agg_arch=args.agg_arch,
                        agg_config=args.agg_config, layer1=args.layer1)
    print(model)

    if args.checkpoint_start is not None:
        print("Start from:", args.checkpoint_start)
        model_state_dict = torch.load(args.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    model = model.eval()
    if use_gpu:
        # model = model.cuda()
        model = model.to(device)

    # Extract feature
    since = time.time()

    query_name = 'query_street'    #1
    gallery_name = 'gallery_satellite'   #1

    which_gallery = which_view(gallery_name)
    which_query = which_view(query_name)

    gallery_path = image_datasets[gallery_name].imgs
    gallery_label, gallery_path  = get_SatId_160k(gallery_path)

    print('%d -> %d:'%(which_query, which_gallery))
    
    with torch.no_grad():
        print('-------------------extract query feature----------------------')
        query_feature = extract_feature(model, dataloaders[query_name])
        print('-------------------extract gallery feature----------------------')
        gallery_feature = extract_feature(model,dataloaders[gallery_name])        
        print('--------------------------ending extract-------------------------------')

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    save_filename = 'answer.txt'
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    results_rank10 = []
    print(len(query_feature))
    gallery_label = np.array(gallery_label)
    for i in range(len(query_feature)):
        result_rank10 = get_result_rank10(query_feature[i], gallery_feature, gallery_label)
        results_rank10.append(result_rank10)

    results_rank10 = np.vstack(results_rank10)
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    with open(save_filename, 'w') as f:
        for row in results_rank10:
            f.write('\t'.join(map(str, row)) + '\n')
