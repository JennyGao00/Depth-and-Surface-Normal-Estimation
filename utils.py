from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
import scipy.spatial
import matplotlib.pyplot as plt
from PIL import Image

# Depth Intrinsic Parameters
# fx_d = 5.8262448167737955e+02
# fy_d = 5.8269103270988637e+02
fx_d = 519.0
fy_d = 519.0
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02


# PCA_utils
def compute_cov_matrices_dense(pos, dense_adj, row):
    col = dense_adj.view(-1)
    (N, D), E = pos.size(), (dense_adj.size(0) * dense_adj.size(1))
    d = (pos[col] - pos[row]).view(N, -1, 3)
    # center
    centers = d.mean(1)
    d = d - centers.view(-1, 1, 3)
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D)).view(N, -1, 3, 3)
    cov = cov.sum(1)
    return cov


def compute_weighted_cov_matrices_dense(pos, weights, dense_adj, row):
    col = dense_adj.view(-1)  # 3300000
    (N, D), E = pos.size(), (dense_adj.size(0) * dense_adj.size(1))  # 100000, 3, 3300000
    d = (pos[col] - pos[row]).view(N, -1, 3)  # 100000, 33, 3
    # center
    weights_sum = weights.view(N, -1).sum(1)  # 100000; WEIGHT 100000, 33
    centers = (d * weights.view(N, -1, 1)).sum(1) / weights_sum.view(N, 1)  # 100000, 3
    d = d - centers.view(-1, 1, 3)   # 100000, 33, 3
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D)).view(N, -1, 3, 3)  # 100000, 33, 3, 3
    cov = cov * weights.view(N, -1, 1, 1)  # 100000, 33, 3, 3
    cov = cov.sum(1)  # 100000, 3, 3
    return cov


def compute_weighted_cov_matrices(pos, weights, edge_idx):
    row, col = edge_idx
    (N, D), E = pos.size(), row.size(0)
    d = pos[col] - pos[row]
    # center
    weights_sum = scatter_add(weights, row, dim=0, dim_size=N)
    centers = scatter_add(d * weights.view(-1, 1), row, dim=0, dim_size=N) / weights_sum.view(-1, 1)

    d = d - centers[row]
    cov = torch.matmul(d.view(E, D, 1), d.view(E, 1, D))
    cov = cov * weights.view(-1, 1, 1)
    cov = scatter_add(cov, row, dim=0, dim_size=N)
    return cov


def cangle(vec1, vec2):
    n = vec1.norm(p=2, dim=-1)*vec2.norm(p=2, dim=-1)
    mask = (n < 1e-8).float()
    cang = (1-mask)*(vec1*vec2).sum(-1)/(n+mask)
    return cang


def compute_prf(pos, normals, edge_idx):
    row, col = edge_idx
    d = pos[col] - pos[row]
    normals1 = normals[row]
    normals2 = normals[col]
    ppf = torch.stack([cangle(normals1, d), cangle(normals2, d),
                       cangle(normals1, normals2), torch.sqrt((d**2).sum(-1))], dim=-1)
    return ppf

# Point Cloud

def depth2point(depth):
    image = np.squeeze(depth)  # (240, 320)
    x = np.arange(1, image.shape[1]+1)
    y = np.arange(1, image.shape[0]+1)
    xx, yy = np.meshgrid(x, y)
    X = (xx - cx_d) * image / fx_d
    Y = (yy - cy_d) * image / fy_d
    Z = image
    pos = np.stack([X, Y, Z], axis=-1)  # (240, 320, 3)
    pos = pos.astype(np.float32)
    pos = torch.from_numpy(pos)
    pos = torch.unsqueeze(pos, 0)  # (1, 240, 320, 3)
    # pos = pos.permute(0, 3, 1, 2)  # (1, 3, 240, 320)

    return pos


# def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
#     if batch_x is None:
#         batch_x = x.new_zeros(x.size(0), dtype=torch.long)
#
#     if batch_y is None:
#         batch_y = y.new_zeros(y.size(0), dtype=torch.long)
#
#     x = x.view(-1, 1) if x.dim() == 1 else x
#     y = y.view(-1, 1) if y.dim() == 1 else y
#
#     assert x.dim() == 2 and batch_x.dim() == 1
#     assert y.dim() == 2 and batch_y.dim() == 1
#     assert x.size(1) == y.size(1)
#     assert x.size(0) == batch_x.size(0)
#     assert y.size(0) == batch_y.size(0)
#
#     rows = []
#     cols = []
#     denses = []
#     current_first = 0
#
#     for element in torch.unique(batch_x, sorted=True):
#         x_element = x[batch_x == element]
#         y_element = y[batch_y == element]
#         tree = scipy.spatial.cKDTree(x_element)
#         _, col = tree.query(y_element, k=max_num_neighbors, distance_upper_bound=r + 1e-8, eps=1e-8)  # [76800, 13] the coordinate of the 12 neighber points
#         col = [torch.tensor(c) for c in col]  # 78600, 13
#         row = [torch.full_like(c, i) for i, c in enumerate(col)]  # list[78600] ; 78600, 13
#         row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)  # [78600*13]
#
#         # Hack: Fill missing neighbors with self loops, if there are any
#         # missing = (col == tree.n).nonzero()
#         col[col == tree.n] = row[col == tree.n]
#         dense = col.view(-1, max_num_neighbors)
#         col = col + current_first
#         row = row + current_first
#         dense = dense + current_first
#         current_first += x_element.size(0)
#         rows.append(row)
#         cols.append(col)
#         denses.append(dense)
#
#     row = torch.cat(rows, dim=0)
#     col = torch.cat(cols, dim=0)
#     dense = torch.cat(denses, dim=0)
#     return torch.stack([row, col], dim=0), dense  # , missing

#
# def radius_graph(x, r, batch=None, max_num_neighbors=32):
#     return radius(x, x, r, batch, batch, max_num_neighbors + 1)

cmap = plt.cm.viridis
def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row_with_gt(input, depth_gt, normal_gt, coarse_normal, refine_normal):
    rgb_cpu = np.squeeze(input.detach().cpu().numpy())+0.5
    rgb = 255 * np.transpose(rgb_cpu, (1,2,0)) # H, W, C

    depth_input_cpu = np.squeeze(depth_gt.detach().cpu().numpy())

    normal_gt_cpu = 0.5*(np.squeeze(normal_gt.detach().cpu().numpy())+1)
    normal_gt = 255 * np.transpose(normal_gt_cpu, (1, 2, 0))  # H, W, C

    coarse_cpu = 0.5 * (np.squeeze(coarse_normal.detach().cpu().numpy()) +1)
    coarse_normal = 255 * np.transpose(coarse_cpu, (1, 2, 0))  # H, W, C
    temp = coarse_normal.copy()
    coarse_normal[:, :, 0] = temp[:, :, 2]
    coarse_normal[:, :, 1] = temp[:, :, 1]
    coarse_normal[:, :, 2] = temp[:, :, 0]

    refine_cpu = 0.5*(np.squeeze(refine_normal.detach().cpu().numpy())+1)
    refine_normal = 255 * np.transpose(refine_cpu, (1, 2, 0))  # H, W, C
    temp = refine_normal.copy()
    refine_normal[:, :, 0] = temp[:, :, 2]
    refine_normal[:, :, 1] = temp[:, :, 1]
    refine_normal[:, :, 2] = temp[:, :, 0]

    d_min = np.min(depth_input_cpu)
    d_max = np.max(depth_input_cpu)
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, normal_gt, coarse_normal, refine_normal])

    return img_merge


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
