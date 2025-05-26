from __future__ import division
import csv
import os
import numpy as np
import pandas as pd
import torch
from scipy import io
import random
import math
from torch.nn.parameter import Parameter
import torch.nn as nn

class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        return (self.data_set['ID'], self.data_set['IM'],
                self.data_set['mc'][index]['train'], self.data_set['mc'][index]['test'],
                self.data_set['mc_p'], self.data_set['mc_true'],
                self.data_set['independent'][0]['train'], self.data_set['independent'][0]['test'])

    def __len__(self):
        return self.nums


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        mc_data = [[float(i) for i in row] for row in reader]
        return torch.FloatTensor(mc_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        mc_data = [[float(i) for i in row.split()] for row in reader]
        return torch.FloatTensor(mc_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = torch.FloatTensor(matrix[name])
    return matrix


def read_mc_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_') + 1:filename.index('.') - 1]
        num = int(filename[filename.index('.') - 1])
        result[num - 1][data_type] = read_csv(os.path.join(path, filename))
    return result


def prepare_data(opt):
    dataset = {}
    cc_data = pd.read_csv(os.path.join(opt.data_path, 'circ_fusion_9905.csv'), index_col=0)
    cc_mat = np.array(cc_data)
    mm_data = pd.read_csv(os.path.join(opt.data_path, 'mi_fusion_9905.csv'), index_col=0)
    mm_mat = np.array(mm_data)
    mi_ci_data = pd.read_csv(os.path.join(opt.data_path, 'cmi_9905.csv'), index_col=0)
    dataset['mc_p'] = torch.FloatTensor(np.array(mi_ci_data))
    dataset['mc_true'] = dataset['mc_p']
    all_zero_index = []
    all_one_index = []
    for i in range(dataset['mc_p'].size(0)):
        for j in range(dataset['mc_p'].size(1)):
            if dataset['mc_p'][i][j] < 1:
                all_zero_index.append([i, j])
            else:
                all_one_index.append([i, j])
    np.random.seed(0)
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)
    zero_tensor = torch.LongTensor(all_zero_index)
    zero_index = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)
    one_tensor = torch.LongTensor(all_one_index)
    one_index = one_tensor.split(int(one_tensor.size(0) / 10), dim=0)
    cross_zero_index = torch.cat([zero_index[i] for i in range(9)])
    cross_one_index = torch.cat([one_index[j] for j in range(9)])
    new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)
    new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)
    dataset['mc'] = []
    for i in range(opt.validation):
        a = [i for i in range(opt.validation)]
        if opt.validation != 1:
            del a[i]
        dataset['mc'].append({'test': [new_one_index[i], new_zero_index[i]],
                              'train': [torch.cat([new_one_index[j] for j in a]),
                                        torch.cat([new_zero_index[j] for j in a])]})

    dataset['independent'] = []
    in_zero_index_test = zero_index[-2]
    in_one_index_test = one_index[-2]
    dataset['independent'].append({'test': [in_one_index_test, in_zero_index_test],
                                   'train': [cross_one_index, cross_zero_index]})

    nd = mi_ci_data.shape[1]
    nm = mi_ci_data.shape[0]
    ID = np.zeros([nd, nd])
    for h1 in range(nd):
        for h2 in range(nd):
                ID[h1, h2] = cc_mat[h1,h2]
    IM = np.zeros([nm, nm])
    for q1 in range(nm):
        for q2 in range(nm):
                IM[q1, q2] = mm_mat[q1, q2]
    dataset['ID'] = torch.from_numpy(ID)
    dataset['IM'] = torch.from_numpy(IM)
    return dataset


def prepare_inputs(fold_data, device):
    ci_sim_integrate_tensor = fold_data[0].to(device)
    mi_sim_integrate_tensor = fold_data[1].to(device)
    mc_p_matrix = fold_data[4].numpy()


    from hypergraph_utils import constructHW_highorder_bipartite
    G_mi, G_ci = constructHW_highorder_bipartite(mc_p_matrix)
    G_mi = G_mi.to(device)
    G_ci = G_ci.to(device)


    concat_miRNA = np.hstack([mc_p_matrix, mi_sim_integrate_tensor.detach().cpu().numpy()])
    concat_mi_tensor = torch.FloatTensor(concat_miRNA).to(device)

    concat_ci = np.hstack([mc_p_matrix.T, ci_sim_integrate_tensor.detach().cpu().numpy()])
    concat_ci_tensor = torch.FloatTensor(concat_ci).to(device)

    return concat_mi_tensor, concat_ci_tensor, G_mi, G_ci


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def StorFile(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target, alpha):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1 - alpha) * loss_sum[one_index].sum() + alpha * loss_sum[zero_index].sum()


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def Eu_dis(x):
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat
