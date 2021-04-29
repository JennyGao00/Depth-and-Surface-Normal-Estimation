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

def change_channel(outputs_norm, scale=1):
    row,col,channel=outputs_norm.shape
    nx = np.ones((row,col)) -outputs_norm[:, :, 0]
    ny = np.ones((row,col)) - outputs_norm[:, :, 1]
    nz = outputs_norm[:, :, 2]
    new_norm = [nx, nz, ny]
    return new_norm
def get_dataList(filename):
    f = open(filename, 'r')
    data_list = list()
    while 1:
        line = f.readline()
        line = line.strip()
        if (not line):
            break
        data_list.append(line)
    f.close()
    return data_list

def load_resume_state_dict(model, resume_state_dict):
     
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    resume_state_dict = {k: v for k, v in resume_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(resume_state_dict)

    return model_dict 

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]
        
def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module'
        new_state_dict[name] = v
    return new_state_dict

def norm_tf(outputs):
    bz, ch, img_rows, img_cols = outputs.size()
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)
    outputs_n = 0.5*(outputs_n+1)                
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n

def norm_sm(outputs):
    bz, ch, img_rows, img_cols = outputs.size()
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)              
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n

def norm_imsave(outputs):
    # outputs_s = np.squeeze(outputs.data.cpu().numpy(), axis=0)
    # outputs_s = outputs_s.transpose(1, 2, 0)
    # outputs_s = outputs_s.reshape(-1,3)
    # outputs_norm = sk.normalize(outputs_s, norm='l2', axis=1)
    # outputs_norm = outputs_norm.reshape(orig_size[0], orig_size[1], 3)
    # outputs_norm = 0.5*(outputs_norm+1)
    bz, ch, img_rows, img_cols = outputs.size()# bz should be one for imsave
    outputs = outputs.permute(0,2,3,1).contiguous().view(-1,ch)
    outputs_n = F.normalize(outputs,p=2)
    outputs_n = 0.5*(outputs_n+1)                
    outputs_n = outputs_n.view(-1, img_rows, img_cols, ch)
    # outputs_n = outputs_n.permute(0,3,1,2)

    return outputs_n


def get_fconv_premodel(model_F, resume_state_dict):
    model_params = model_F.state_dict()

    # copy parameter from resume_state_dict
    # conv1, conv+bn+conv+bn
    model_params['module.conv1.conv.0.weight'] = resume_state_dict['module.conv1.conv.0.weight']
    model_params['module.conv1.conv.0.bias'] = resume_state_dict['module.conv1.conv.0.bias']
    model_params['module.conv1.conv.1.weight'] = resume_state_dict['module.conv1.conv.1.weight']
    model_params['module.conv1.conv.1.bias'] = resume_state_dict['module.conv1.conv.1.bias']  
    model_params['module.conv1.conv.3.weight'] = resume_state_dict['module.conv1.conv.3.weight']
    model_params['module.conv1.conv.3.bias'] = resume_state_dict['module.conv1.conv.3.bias']
    model_params['module.conv1.conv.4.weight'] = resume_state_dict['module.conv1.conv.4.weight']
    model_params['module.conv1.conv.4.bias'] = resume_state_dict['module.conv1.conv.4.bias']  

    # conv2, conv+bn+conv+bn
    model_params['module.conv2.conv.0.weight'] = resume_state_dict['module.conv2.conv.0.weight']
    model_params['module.conv2.conv.0.bias'] = resume_state_dict['module.conv2.conv.0.bias']
    model_params['module.conv2.conv.1.weight'] = resume_state_dict['module.conv2.conv.1.weight']
    model_params['module.conv2.conv.1.bias'] = resume_state_dict['module.conv2.conv.1.bias']  
    model_params['module.conv2.conv.3.weight'] = resume_state_dict['module.conv2.conv.3.weight']
    model_params['module.conv2.conv.3.bias'] = resume_state_dict['module.conv2.conv.3.bias']
    model_params['module.conv2.conv.4.weight'] = resume_state_dict['module.conv2.conv.4.weight']
    model_params['module.conv2.conv.4.bias'] = resume_state_dict['module.conv2.conv.4.bias']  

    # conv3, conv+bn+conv+bn+conv+bn
    model_params['module.conv3.conv.0.weight'] = resume_state_dict['module.conv3.conv.0.weight']
    model_params['module.conv3.conv.0.bias'] = resume_state_dict['module.conv3.conv.0.bias']
    model_params['module.conv3.conv.1.weight'] = resume_state_dict['module.conv3.conv.1.weight']
    model_params['module.conv3.conv.1.bias'] = resume_state_dict['module.conv3.conv.1.bias']  
    model_params['module.conv3.conv.3.weight'] = resume_state_dict['module.conv3.conv.3.weight']
    model_params['module.conv3.conv.3.bias'] = resume_state_dict['module.conv3.conv.3.bias']
    model_params['module.conv3.conv.4.weight'] = resume_state_dict['module.conv3.conv.4.weight']
    model_params['module.conv3.conv.4.bias'] = resume_state_dict['module.conv3.conv.4.bias'] 
    model_params['module.conv3.conv.6.weight'] = resume_state_dict['module.conv3.conv.6.weight']
    model_params['module.conv3.conv.6.bias'] = resume_state_dict['module.conv3.conv.6.bias']
    model_params['module.conv3.conv.7.weight'] = resume_state_dict['module.conv3.conv.7.weight']
    model_params['module.conv3.conv.7.bias'] = resume_state_dict['module.conv3.conv.7.bias']   
    return model_params


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

def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    rows = []
    cols = []
    denses = []
    current_first = 0

    for element in torch.unique(batch_x, sorted=True):
        x_element = x[batch_x == element]
        y_element = y[batch_y == element]
        tree = scipy.spatial.cKDTree(x_element)
        _, col = tree.query(y_element, k=max_num_neighbors, distance_upper_bound=r + 1e-8, eps=1e-8)  # [76800, 13] the coordinate of the 12 neighber points
        col = [torch.tensor(c) for c in col]  # 78600, 13
        row = [torch.full_like(c, i) for i, c in enumerate(col)]  # list[78600] ; 78600, 13
        row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)  # [78600*13]

        # Hack: Fill missing neighbors with self loops, if there are any
        # missing = (col == tree.n).nonzero()
        col[col == tree.n] = row[col == tree.n]
        dense = col.view(-1, max_num_neighbors)
        col = col + current_first
        row = row + current_first
        dense = dense + current_first
        current_first += x_element.size(0)
        rows.append(row)
        cols.append(col)
        denses.append(dense)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    dense = torch.cat(denses, dim=0)
    return torch.stack([row, col], dim=0), dense  # , missing


cmap = plt.cm.viridis


def radius_graph(x, r, batch=None, max_num_neighbors=32):
    return radius(x, x, r, batch, batch, max_num_neighbors + 1)


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

    refine_cpu = 0.5*(np.squeeze(refine_normal.detach().cpu().numpy())+1)
    refine_normal = 255 * np.transpose(refine_cpu, (1, 2, 0))  # H, W, C

    d_min = np.min(depth_input_cpu)
    d_max = np.max(depth_input_cpu)
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, normal_gt, coarse_normal, refine_normal])

    return img_merge


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
