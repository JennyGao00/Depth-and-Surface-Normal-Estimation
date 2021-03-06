# depth estimation from a single image, then calculate the point normal
# and extract point feature from point cloud converted from the predicted depth.

# Implemented in Pytorch by Gao jinyan, 20210409

import torch.nn as nn
import torch
from models.models_utils import *
from models.PointModel import PointModel
from utils import depth2point


# class Normalize(nn.Module):
#     def __init__(self):
#         super(Normalize, self).__init__()
#
#     def forward(self, bottom):
#         qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
#         top = bottom.div(qn)
#
#         return top

class DepthModel(nn.Module):
    def __init__(self, input_channel, output_channel, track_running_static=True):
        super(DepthModel, self).__init__()
        # self.input_channel = input_channel
        # self.output_channel = output_channel
        # self.track = track_running_static
        # filters = [64, 128, 256, 256]
        #
        # # encoder
        # self.conv1 = create_conv_2(self.input_channel, filters[0], track=self.track)
        # self.conv2 = create_conv_2(filters[0], filters[1], track=self.track)
        # self.conv3 = create_conv_2(filters[1], filters[2], track=self.track)
        # self.conv4 = create_conv_2(filters[2], filters[3], track=self.track)
        #
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #
        # # decoder
        # self.deconv4 = create_deconv_2(filters[3], filters[2], track=self.track)
        # self.deconv3 = create_deconv_2(filters[2] + filters[2], filters[1], track=self.track)
        # self.deconv2 = create_deconv_2(filters[1] + filters[1], filters[0], track=self.track)
        # self.deconv1 = create_addon(filters[0] + filters[0], filters[0], self.output_channel)
        #
        # # Use bilinear unpooling instead of max-pooling to avoid blocky phenomenon
        # # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.pointNet = PointModel()
        # self.normalize = Normalize()

    def forward(self, output_depth):
        # features1 = self.conv1(input)
        # features1_p, indices1_p = self.pool1(features1)
        # features2 = self.conv2(features1_p)
        # features2_p, indices2_p = self.pool2(features2)
        # features3 = self.conv3(features2_p)
        # features3_p, indices3_p = self.pool3(features3)
        # features4 = self.conv4(features3_p)
        #
        # defeature3t = self.deconv4(features4)
        # defeature3 = torch.cat((self.unpool3(defeature3t), features3), 1)
        # defeature2t = self.deconv3(defeature3)
        # defeature2 = torch.cat((self.unpool2(defeature2t), features2), 1)
        # defeature1t = self.deconv2(defeature2)
        # defeature1 = torch.cat((self.unpool1(defeature1t), features1), 1)
        #
        # output_depth = self.deconv1(defeature1)

        point = depth2point(output_depth.detach().cpu().numpy())
        pca_normal, point_fea = self.pointNet(point)
        # pca_normal = self.normalize(pca_normal)

        return pca_normal, point_fea

