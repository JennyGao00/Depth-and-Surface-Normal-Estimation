# input => coarse normal, point normal, point feature.
#output => refined normal.
# Implemented in Pytorch by Gao Jinyan, 20210409

import torch.nn as nn
import torch
from models.models_utils import *
from torch.nn import Sequential as S, Linear as L, ReLU
from torch_geometric.nn.inits import reset
from collections import OrderedDict



class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top

#
# class FTB_block(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
#         self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
#         self.bn1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=False)
#
#     def forward(self, coarse, pca):
#         coarse_normal = coarse.contiguous() + pca.contiguous()
#
#         x = self.conv1(coarse_normal)
#         residual = x
#         out = self.conv2(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class AFA_block(nn.Module):
#     def __init__(self, dim_in, dim_out): # 16, 3
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.dim_mid = 13
#         self.globalpool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(self.dim_in, self.dim_mid, 1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(self.dim_mid, self.dim_out, 1, stride=1, padding=0, bias=False)
#         self.sigmd = nn.Sigmoid()
#
#     def forward(self, normal, point_fea):
#         w = torch.cat([normal, point_fea], 1)
#         w = self.globalpool(w)
#         w = self.conv1(w)
#         w = self.relu(w)
#         w = self.sigmd(w)
#         w = w * point_fea
#         w = self.conv2(w)
#         # w = self.relu(w)
#         out = w.contiguous() + normal.contiguous()
#         return out


# class AFA_block(nn.Module):
#     def __init__(self, dim_in, dim_out): # 6, 3
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         # self.dim_mid = 13
#         # self.globalpool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 1, stride=1, padding=0, bias=False)
#         self.sigmd = nn.Sigmoid()
#         # self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, coarse, pca):
#         w = torch.cat([coarse, pca], 1)  # 1, 6, 256, 256
#         # w = self.globalpool(w)
#         w = self.conv1(w)
#         # w = self.relu(w)
#         w = self.sigmd(w)
#         w = w * pca
#         w = self.conv2(w)
#         w = self.relu(w)
#         out = w.contiguous() + coarse.contiguous()
#         return out


# class FTB_block(nn.Module):
#     def __init__(self, dim_in, dim_out):  # 16, 3
#         super().__init__()
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.conv1 = nn.Conv2d(self.dim_in, self.dim_out, 1, stride=1, padding=0, bias=False)
#         self.conv2 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=True)
#         self.bn1 = nn.BatchNorm2d(self.dim_out, momentum=0.5)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(self.dim_out, self.dim_out, 3, stride=1, padding=2, dilation=2, bias=False)
#
#     def forward(self, normal, fea):
#         x = self.conv1(torch.cat([normal, fea], 1))
#         out = self.conv2(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out += normal
#         # out = self.relu(out)
#         return out
#

class NormalModel(nn.Module):
    
    def __init__(self, input_channel, output_channel, track_running_static=True):
        super(NormalModel, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.track = track_running_static
        filters = [64, 128, 256, 256]
        filters_fconv = [24, 16, 8]

        # encoder
        self.conv1 = create_conv_2(self.input_channel, filters[0], track=self.track)
        self.conv2 = create_conv_2(filters[0], filters[1], track=self.track)
        self.conv3 = create_conv_2(filters[1], filters[2], track=self.track)
        self.conv4 = create_conv_2(filters[2], filters[3], track=self.track)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # decoder
        self.deconv4 = create_deconv_2(filters[3], filters[2], track=self.track)
        self.deconv3 = create_deconv_2(filters[2] + filters[2], filters[1], track=self.track)
        self.deconv2 = create_deconv_2(filters[1] + filters[1], filters[0], track=self.track)
        self.deconv1 = create_addon(filters[0] + filters[0], filters[0], self.output_channel)

        # Use bilinear unpooling instead of max-pooling to avoid blocky phenomenon
        # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.normalize = Normalize()

        # method1
        # self.coarse_pca = AFA_block(6,3)
        # self.refine_point = FTB_block(16,3)

        # method2
        # self.P_model = nn.Sequential(
        #           nn.Conv2d(3, 16, 3, padding=1, dilation=1),
        #           nn.ReLU(),
        #           nn.Conv2d(16, 8, 3, padding=1, dilation=1),
        #           nn.ReLU(),
        #           nn.Conv2d(8, 3, 3, padding=1, dilation=1),
        #           nn.ReLU()
        #         )
        #
        # # encoder in the final stage
        # self.conv1_f1 = create_conv_2_in(16, filters_fconv[0],
        #                                  track=self.track)  # 16->32->32
        # self.conv2_f1 = create_conv_2_in(filters_fconv[0], filters_fconv[1], track=self.track)  # 32->16->16
        # # decoder in the final stage
        # self.deconv1_f1 = create_addon(filters_fconv[1], filters_fconv[2], self.output_channel)  # 16->8->3

        # method3
        self.P_model = nn.Sequential(
            nn.Conv2d(16, 12, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(12, 6, 3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(6, 3, 3, padding=1, dilation=1),
            nn.ReLU()
        )

        # encoder in the final stage
        self.conv1_f1 = create_conv_2_in(6, filters_fconv[0],
                                         track=self.track)  # 16->32->32
        self.conv2_f1 = create_conv_2_in(filters_fconv[0], filters_fconv[1], track=self.track)  # 24->16->16
        # decoder in the final stage
        self.deconv1_f1 = create_addon(filters_fconv[1], filters_fconv[2], self.output_channel)  # 16->8->3



    def forward(self, input, pca_normal, point_fea):
        features1 = self.conv1(input)
        features1_p, indices1_p = self.pool1(features1)
        features2 = self.conv2(features1_p)
        features2_p, indices2_p = self.pool2(features2)
        features3 = self.conv3(features2_p)
        features3_p, indices3_p = self.pool3(features3)
        features4 = self.conv4(features3_p)

        defeature3t = self.deconv4(features4)
        defeature3 = torch.cat((self.unpool3(defeature3t), features3), 1)
        defeature2t = self.deconv3(defeature3)
        defeature2 = torch.cat((self.unpool2(defeature2t), features2), 1)
        defeature1t = self.deconv2(defeature2)
        defeature1 = torch.cat((self.unpool1(defeature1t), features1), 1)

        coarse_normal = self.deconv1(defeature1)
        coarse_normal = self.normalize(coarse_normal)  # unet_coarse_normal

        # method1
        # normal = self.coarse_pca(coarse_normal, pca_normal)
        # refine_normal = self.refine_point(normal, point_fea)
        # refine_normal = self.normalize(refine_normal)

        # method2
        # coarse_normal = coarse_normal.contiguous() + pca_normal.contiguous()
        # normal = self.P_model(coarse_normal)
        # fea = torch.cat((normal, point_fea), 1)
        # features1 = self.conv1_f1(fea)
        # features2 = self.conv2_f1(features1)
        # refine_normal = self.deconv1_f1(features2)
        # refine_normal = self.normalize(refine_normal)

        # method3
        p_fea = torch.cat((pca_normal, point_fea), 1)  # 16
        point = self.P_model(p_fea)  # 16=>3

        fea = torch.cat((coarse_normal, point), 1)  # 6
        features1 = self.conv1_f1(fea)
        features2 = self.conv2_f1(features1)
        refine_normal = self.deconv1_f1(features2)
        refine_normal = self.normalize(refine_normal)



        return coarse_normal, refine_normal
        



