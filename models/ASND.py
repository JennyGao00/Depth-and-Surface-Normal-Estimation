import torch
import torch.nn as nn
import torch.nn.functional as F
from models.task_specific_layers import TaskBasicBlock, TaskConv2d, TaskBatchNorm2d, TaskSABlock

import pdb
from termcolor import colored
from depth_normal_fuse.depth2normal_light import Depth2normalLight


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class ScalePredictionModule(nn.Module):
    """ Module to make the inital task predictions """

    def __init__(self, input_channels, task_channels, task):
        super(ScalePredictionModule, self).__init__()

        # Per task feature refinement + decoding
        if input_channels == task_channels:
            channels = input_channels
            self.refinement = nn.Sequential(TaskBasicBlock(channels, channels, task=task),
                                            TaskBasicBlock(channels, channels, task=task))

        else:

            downsample = nn.Sequential(
                # add feature and task estimation
                TaskConv2d(input_channels, task_channels, 1, bias=False,
                           task=task),
                TaskBatchNorm2d(task_channels, task=task))
            self.refinement = nn.Sequential(
                TaskBasicBlock(input_channels, task_channels,
                               downsample=downsample, task=task),
                TaskBasicBlock(task_channels, task_channels, task=task))

    def forward(self, features_curr_scale, features_prev_scale=None):
        if features_prev_scale is not None:  # Concat features that were propagated from previous scale
            x = torch.cat(
                (features_curr_scale,
                 F.interpolate(features_prev_scale, scale_factor=2, mode='bilinear'),  # features
                 ), 1)

        else:
            x = features_curr_scale

        # Refinement
        out = self.refinement(x)

        return out


class DepthLayer(nn.Module):
    def __init__(self, input_channels):
        super(DepthLayer, self).__init__()

        self.pred = TaskConv2d(input_channels, 1, 1, task="depth")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pred(x)
        x = self.sigmoid(x)

        return x


class NormalLayer(nn.Module):
    def __init__(self, input_channels):
        super(NormalLayer, self).__init__()

        self.pred = TaskConv2d(input_channels, 3, 1, task="normals")
        self.normalize = Normalize()

    def forward(self, x):
        x = self.pred(x)
        x = self.normalize(x)

        return x


def compute_kernel(input_for_kernel, input_mask=None, kernel_size=3, stride=1, padding=1, dilation=1,
                   kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=True):
    return kernel2d(input_for_kernel, input_mask,
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    dilation=dilation, kernel_type=kernel_type,
                    smooth_kernel_type=smooth_kernel_type,
                    smooth_kernel=None,
                    inv_alpha=None,
                    inv_lambda=None,
                    channel_wise=False,
                    normalize_kernel=normalize_kernel,
                    transposed=False,
                    native_impl=None)


class GeoDepthNet(nn.Module):
    """
    Depth estimation Network with pixel adaptive surface normal constraint
    output 1/2 size estimated depth compared with input image
    """

    def __init__(self, p, backbone, backbone_channels, max_depth=10.):
        super(GeoDepthNet, self).__init__()
        # General
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.num_scales = len(backbone_channels)

        if 'resnet' in p['backbone']:
            self.task_channels = [x // 4 for x in backbone_channels]
        else:
            self.task_channels = backbone_channels
        print("task_channels: ", self.task_channels)
        self.channels = backbone_channels
        self.max_depth = p['max_depth'] if 'max_depth' in p.keys() else max_depth

        self.use_gt_depth = p['use_gt_depth'] if 'use_gt_depth' in p.keys() else False
        self.use_guidance = p['use_guidance'] if 'use_guidance' in p.keys() else False
        self.guidance_reduce = p['guidance_reduce'] if 'guidance_reduce' in p.keys() else False

        self.normal_loss = p['normal_loss'] if 'normal_loss' in p.keys() else False

        # Backbone
        self.backbone = backbone

        # Initial task predictions at multiple scales
        ################################# Depth branch #############################################
        self.scale_0_fea_depth = ScalePredictionModule(self.channels[0] + self.task_channels[1] + 1 * 1,
                                                       self.task_channels[0], task='depth')
        self.scale_0_depth = DepthLayer(self.task_channels[0])
        self.scale_1_fea_depth = ScalePredictionModule(self.channels[1] + self.task_channels[2] + 1 * 1,
                                                       self.task_channels[1], task='depth')
        self.scale_1_depth = DepthLayer(self.task_channels[1])
        self.scale_2_fea_depth = ScalePredictionModule(self.channels[2] + self.task_channels[3] + 1 * 1,
                                                       self.task_channels[2], task='depth')
        self.scale_2_depth = DepthLayer(self.task_channels[2])
        self.scale_3_fea_depth = ScalePredictionModule(self.channels[3],
                                                       self.task_channels[3], task='depth')
        self.scale_3_depth = DepthLayer(self.task_channels[3])

        ################################# Guidance branch #############################################
        if self.use_guidance:
            self.scale_0_guidance = ScalePredictionModule(self.channels[0] + self.task_channels[1],
                                                          self.task_channels[0], task='guidance')

            if self.guidance_reduce:
                self.scale_0_guidance_reduce_dims = nn.Sequential(TaskConv2d(self.task_channels[0], 3, 1, bias=True,
                                                                             task='guidance'),
                                                                  torch.nn.Sigmoid())  # easy to visualize
            self.scale_1_guidance = ScalePredictionModule(self.channels[1] + self.task_channels[2],
                                                          self.task_channels[1], task='guidance')
            self.scale_2_guidance = ScalePredictionModule(self.channels[2] + self.task_channels[3],
                                                          self.task_channels[2], task='guidance')
            self.scale_3_guidance = ScalePredictionModule(self.channels[3],
                                                          self.task_channels[3], task='guidance')

        # Depth Normal conversion modules
        k_size = p['k_size'] if 'k_size' in p.keys() else 5
        sample_num = p['sample_num'] if 'sample_num' in p.keys() else 40
        self.scale_0_conversion = DepthNormalConversion(k_size=k_size, dilation=1,
                                                        sample_num=sample_num)  # Depth2NormalLight

    def set_task_specific_parameters_trainable(self, task):
        for param in self.parameters():
            param.requires_grad = False  # first set all parameters to False

        # now set task-specific parameters to True
        for m in self.modules():
            if isinstance(m, TaskConv2d) or isinstance(m, TaskBatchNorm2d) or isinstance(m, PacConv2d):
                if m.task == task:
                    for param in m.parameters():
                        param.requires_grad = True

        for param in self.heads[task].parameters():
            param.requires_grad = True

    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(colored(name, 'green'))
        print("-----------------------------------------------------------------")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(colored(name, 'magenta'))

    def scale_intrinsic(self, intrinsic, scale_x, scale_y):
        intrinsic = intrinsic.clone()
        intrinsic[:, 0, :] = intrinsic[:, 0, :] * scale_x
        intrinsic[:, 1, :] = intrinsic[:, 1, :] * scale_y

        # print(scale_x, scale_y)
        # print(intrinsic)
        return intrinsic

    def forward(self, x, intrinsic, gt_depth=None):
        # pdb.set_trace()

        img_size = x.size()[-2:]
        img_H, img_W = img_size[0], img_size[1]

        # upscale 2x for calculate normal
        scale_factor = 2

        out = {}

        # Backbone
        x = self.backbone(x)

        # Predictions at multiple scales
        # Scale 3
        x_3_fea_depth = self.scale_3_fea_depth(x[3])
        x_3_depth = self.scale_3_depth(x_3_fea_depth) * self.max_depth

        x_2_fea_depth = self.scale_2_fea_depth(x[2], torch.cat([x_3_fea_depth, x_3_depth], dim=1))
        x_2_depth = self.scale_2_depth(x_2_fea_depth) * self.max_depth

        x_1_fea_depth = self.scale_1_fea_depth(x[1], torch.cat([x_2_fea_depth, x_2_depth], dim=1))
        x_1_depth = self.scale_1_depth(x_1_fea_depth) * self.max_depth

        x_0_fea_depth = self.scale_0_fea_depth(
            F.interpolate(x[0], scale_factor=scale_factor, mode='bilinear'),
            F.interpolate(torch.cat([x_1_fea_depth, x_1_depth], dim=1), scale_factor=scale_factor, mode='bilinear'))
        x_0_depth = self.scale_0_depth(x_0_fea_depth) * self.max_depth

        if self.use_guidance:
            x_3_guidance = self.scale_3_guidance(x[3])
            x_2_guidance = self.scale_2_guidance(x[2], x_3_guidance)
            x_1_guidance = self.scale_1_guidance(x[1], x_2_guidance)
            x_0_guidance = self.scale_0_guidance(
                F.interpolate(x[0], scale_factor=scale_factor, mode='bilinear'),
                F.interpolate(x_1_guidance, scale_factor=scale_factor, mode='bilinear')
            )
            x_0_guidance = self.scale_0_guidance_reduce_dims(x_0_guidance)

        else:
            x_0_guidance = None

        # scale intrinsic
        _, _, scale_0_H, scale_0_W = x_0_depth.shape
        scale_0_intrinsic = self.scale_intrinsic(intrinsic, scale_x=scale_0_W / img_W, scale_y=scale_0_H / img_H)

        if self.normal_loss:
            if not self.use_gt_depth or gt_depth is None:
                #             print("not use gt_depth")
                x_0_converted_normals = self.scale_0_conversion(
                    x_0_depth, scale_0_intrinsic, x_0_guidance)
            else:
                #             print("use gt_depth")
                gt_depth = F.interpolate(gt_depth, (scale_0_H, scale_0_W), mode='bilinear')
                x_0_converted_normals = self.scale_0_conversion(
                    gt_depth, scale_0_intrinsic, x_0_guidance)

            x_0_converted_normals = F.interpolate(x_0_converted_normals, img_size, mode='bilinear')
        else:
            x_0_converted_normals = None

        out['guidance_feature'] = F.interpolate(x_0_guidance, img_size,
                                                mode='bilinear') if x_0_guidance is not None else None

        out['deep_supervision'] = {'scale_0': {'depth': x_0_depth},
                                   'scale_1': {'depth': x_1_depth},
                                   'scale_2': {'depth': x_2_depth},
                                   'scale_3': {'depth': x_3_depth}}

        out['converted_normals'] = x_0_converted_normals

        if x_0_converted_normals is not None:
            out['normals'] = x_0_converted_normals

        out['depth'] = F.interpolate(x_0_depth, img_size, mode='bilinear')

        return out