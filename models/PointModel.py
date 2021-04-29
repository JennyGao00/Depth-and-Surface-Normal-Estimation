import torch
from models.gnn import GNNFixedK
from torch_sym3eig import Sym3Eig
from utils import radius_graph, compute_cov_matrices_dense, depth2point, compute_prf
from torch_geometric.nn.inits import reset
from loader import get_data_path, get_loader
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential as S, Linear as L, ReLU
import torch.nn.functional as F
from models.quaternion import QuatToMat
from PIL import Image

class PointModel(torch.nn.Module):
    def __init__(self):
        super(PointModel, self).__init__()
        # self.stepWeights = GNNFixedK()
        # self.dropout = torch.nn.Dropout(p=0.25)
        self.k_size = 12
        self.layer1 = S(L(7, 32), ReLU(), L(32, 16))
        self.layerg = S(L(19, 32), ReLU(), L(32, 8))
        self.layer2 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg2 = S(L(16, 32), ReLU(), L(32, 8))
        self.layer3 = S(L(24, 32), ReLU(), L(32, 16))
        self.layerg3 = S(L(16, 32), ReLU(), L(32, 12))
        self.layer4 = S(L(27, 64), ReLU(), L(64, 1))
        self.layer5 = S(L(13, 64), ReLU(), L(64, 3))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.layer1)
        reset(self.layerg)
        reset(self.layer2)
        reset(self.layerg2)
        reset(self.layer3)
        reset(self.layerg3)
        reset(self.layer4)
        reset(self.layer5)

    def forward(self, points): # pos (1, 240, 320, 3)
        pos = points.view(-1, 3)  #(240*320, 3)

        # Compute KNN-graph indices for GNN
        edge_idx_l, dense_l = radius_graph(pos, 0.5, batch=None, max_num_neighbors=self.k_size)  # [2, 76800*13] [76800, 13]
        # (PCA)
        cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()  # 76800, 3, 3

        # add noise to avoid nan
        noise = (torch.rand(100, 3) - 0.5) * 1e-8  # 100, 3
        cov = cov + torch.diag(noise).cuda()

        eig_val, eig_vec = Sym3Eig.apply(cov)  # 76800, 3; 768000, 3, 3
        # _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        _, argsort = eig_val.sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        # mask = torch.isnan(eig_vec)
        # eig_vec[mask] = 0.0
        normals = eig_vec[:, :, 0].cuda()  # 76800, 3
        pca_normals = torch.reshape(normals, (points.shape[0], points.shape[1], points.shape[2], points.shape[3]))  # (1, 240, 320, 3)
        pca_normals = pca_normals.permute(0, 3, 1, 2)  # (1, 3, 240, 320)


        N = pos.size(0)  # 76800
        K = dense_l.size(1)  # 13
        E = edge_idx_l.size(1)  # 76800*13
        pos = pos.detach().cuda()
        edge_idx_l = edge_idx_l.cuda()
        rows, cols =edge_idx_l   # 76800*13, 76800*13
        cart = pos[cols] - pos[rows]  # 76800*13, 3  // p_j - p_i

        ppf = compute_prf(pos, normals, edge_idx_l)  # 998400, 4

        x = torch.cat([cart, ppf], dim=-1)  # 998400, 7
        x = self.layer1(x)  # 998400, 16  # h
        x = x.view(N, K, -1)  # 76800, 13, 16
        global_x = x.mean(1)  # 76800, 16
        global_x = torch.cat([global_x, normals.view(-1, 3)], dim=-1)  # 76800, 19
        x_g = self.layerg(global_x)  # 76800, 8      # gamma
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)  # 998400, 16+8=24
        x = self.layer2(x)  # 998400, 16             # h
        x = x.view(N, K, -1)  # 76800, 13, 16
        global_x = x.mean(1)  # 76800, 16
        x_g = self.layerg2(global_x)  # 76800, 8     # gamma
        x = torch.cat([x.view(E, -1), x_g[rows]], dim=1)  # 998400, 16+8=24    # gamma
        x = self.layer3(x)  # 998400, 16             # h
        x = x.view(N, K, -1)  # 76800, 13, 16
        global_x = x.mean(1)  # 76800, 16
        x_g = self.layerg3(global_x)  # 76800, 12    # gamma
        quat = x_g[:, :4]  # (76800,4)
        quat = quat / (quat.norm(p=2, dim=-1) + 1e-8).view(-1, 1)  # (76800, 4)
        mat = QuatToMat.apply(quat).view(-1, 3, 3)  # 76800, 3, 3

        # Kernel application
        x_g = x_g[:, 4:]  # (76800, 8)
        rot_cart = torch.matmul(mat.view(-1, 3, 3)[rows], cart.view(-1, 3, 1)).view(-1, 3)  # (998400, 3)
        x = torch.cat([x.view(E, -1), x_g[rows], rot_cart], dim=1)  # 998400, 16+8+3=27
        x = self.layer4(x)  # 998400, 1               # phi
        x = x.view(N, K)  # 76800, 13
        # point_fea = self.layer5(x)  # 76800, 3
        # softmax
        point_fea = F.softmax(x, 1)  # 76800, 13

        point_fea = point_fea.view(points.shape[0], points.shape[1], points.shape[2], -1)  # (1, 240, 320, 13)
        point_fea = point_fea.permute(0, 3, 1, 2)  # # (1, 13, 240, 320)

        return pca_normals, point_fea


if __name__ == '__main__':
    # # data_loader = get_loader('nyuv2')
    # # data_path = get_data_path('nyuv2')
    # # t_loader = data_loader(data_path, split='train', img_size=(240, 320), img_norm=True)
    # # trainloader = data.DataLoader(t_loader, batch_size=1, shuffle=True, num_workers=4)
    model = PointModel()
    device = torch.device('cuda')
    model.to(device)
    model.train()
    #
    # for i, (image, depth, normal) in enumerate(trainloader):
    #     bs = 2
    #
    #     # image = image.cuda()
    #     # depth = depth.cuda()
    #     # normal = normal.cuda()

    depth_path = '/home/gao/depth.png'
    # xyz = Visualization_XYZ(1, 1, (240, 320))
    pred_depth = Image.open(depth_path)
    depth = np.array(pred_depth)
    depth = depth[:, :, 0]
    depth = depth / 255
    depth = depth.astype(np.float32)
    depth = depth.reshape(1, 1, depth.shape[0], depth.shape[1])

    # depth = depth.cpu().detach().numpy()
    point = depth2point(depth)
    # point = point.type_as(torch.float32)  # torch.float64 => 32
    pca_normal, point_fea = model(point)

    # imgs = image.numpy()
    # imgs = np.transpose(imgs, [0, 2, 3, 1])
    # imgs = imgs + 0.5
    #
    # normal = normal.numpy()
    # normal = 0.5 * (normal + 1)
    # normal = np.transpose(normal, [0, 2, 3, 1])

    pca_normal = pca_normal.cpu().numpy()
    pca_normal = 0.5 * (pca_normal + 1)
    pca_normal = np.transpose(pca_normal, [0, 2, 3, 1])

    point_fea = point_fea.cpu().detach().numpy()
    point_fea = 0.5 * (point_fea + 1)
    point_fea = np.transpose(point_fea, [0, 2, 3, 1])

    # normal_mask = normal_mask.numpy()
    # normal_mask = np.repeat(normal_mask[:, :, :, np.newaxis], 3, axis=3)

    # depth = depth.numpy()
    depth = np.transpose(depth, [0, 2, 3, 1])
    depth = np.repeat(depth, 3, axis=3)

    # raw_depth_mask = raw_depth_mask.numpy()
    # raw_depth_mask = np.repeat(raw_depth_mask[:, :, :, np.newaxis], 3, axis=3)

    f, axarr = plt.subplots(2, 2)
    for j in range(2):
        # print(im_name[j])
        # axarr[j][0].imshow(imgs[0])
        axarr[j][0].imshow(depth[0])
        # axarr[j][2].imshow(normal_mask[j])
        # axarr[j][2].imshow(normal[0])
        axarr[j][1].imshow(pca_normal[0])
        # axarr[j][4].imshow(point_fea[0])

    plt.show()
    plt.close()