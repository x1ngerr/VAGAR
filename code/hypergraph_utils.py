import numpy as np
import torch

def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and len(h) != 0:
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.Tensor(G)
        return G


def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def constructHW_highorder_bipartite(mc_matrix):
    m, d = mc_matrix.shape
    H_row = np.zeros((m + d, m))
    for i in range(m):
        H_row[i, i] = 1
        for j in range(d):
            if mc_matrix[i, j] != 0:
                H_row[m + j, i] = 1
    H_col = np.zeros((m + d, d))
    for j in range(d):
        H_col[m + j, j] = 1
        for i in range(m):
            if mc_matrix[i, j] != 0:
                H_col[i, j] = 1
    H = hyperedge_concat(H_row, H_col)
    G_all = _generate_G_from_H(H)
    G_mi = G_all[:m, :m]
    G_ci = G_all[m:, m:]
    return G_mi, G_ci
