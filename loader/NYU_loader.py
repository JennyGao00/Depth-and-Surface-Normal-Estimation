##################################
# load NYU dataset
# JY Gao, 2021
##################################

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from loader.loader_utils import png_reader_32bit, png_reader_uint8

from tqdm import tqdm
from torch.utils import data
from PIL import Image


class nyuLoader(data.Dataset):
    """Data loader for the scannet dataset.

    """

    def __init__(self, root, split, img_size=(240, 320), img_norm=True, mode=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
            else (img_size, img_size)
        self.mode = mode

        for split in ['train', 'test']:
            path = pjoin(self.root, 'nyu_' + split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        # path = pjoin(self.root , 'nyu_' + split + '_list.txt')
        # file_list = tuple(open(path, 'r'))
        # file_list = [id_.rstrip() for id_ in file_list]
        # self.files['train'] = file_list[:458]
        # self.files['test'] = file_list[458:]



    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name_base = self.files[self.split][index]

        # raw_depth
        raw_depth_path = pjoin(self.root, str(im_name_base))
        raw_depth = png_reader_32bit(raw_depth_path, self.img_size)
        raw_depth = raw_depth.astype(float)
        raw_depth = raw_depth / 10000

        # raw_depth_mask
        raw_depth_mask = (raw_depth > 0.0001).astype(float)
        raw_depth = raw_depth[np.newaxis, :, :]
        raw_depth = torch.from_numpy(raw_depth).float()
        raw_depth_mask = torch.from_numpy(raw_depth_mask).float()


        # image
        rgb_path = raw_depth_path.replace('depth', 'colors')
        image = png_reader_uint8(rgb_path, self.img_size)
        image = image.astype(float)
        # image    = image / 255
        image = (image - 128) / 255
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        # normal
        normal_path = raw_depth_path.replace('depth', 'normal')
        normal = png_reader_uint8(normal_path, self.img_size)
        normal = normal.astype(float)
        normal = normal / 255
        normal = normal.transpose(2, 0, 1)

        # normal mask
        # normal_mask = np.power(normal[0], 2) + np.power(normal[1], 2) + np.power(normal[2], 2)
        # normal_mask = (normal_mask > 0.001).astype(float)
        #
        # normal[0][normal_mask == 0] = 0.001
        # normal[1][normal_mask == 0] = 0.001
        # normal[2][normal_mask == 0] = 0.001
        normal = 2 * normal - 1
        normal = torch.from_numpy(normal).float()

        # image          : RGB      3,240,320
        # raw_depth      : depth    1, 240,320
        # raw_depth_mask : 0 or 1,  240,320
        # normal         : /255  *2-1, 3,240,320
        # normal_mask    : 0 or 1,  240,320

        # with masks
        # return image, normal, normal_mask, raw_depth_mask, raw_depth

        # without masks
        return image, raw_depth, normal





if __name__ == '__main__':
    # Config your local data path

    depth_path = '/home/gao/depth.png'

    pred_depth = Image.open(depth_path)
    depth = np.array(pred_depth)
    depth = depth[:, :, 0]
    depth = (depth - 128) / 255
    depth = depth.astype(np.float32)
    depth = depth.reshape(1, 1, depth.shape[0], depth.shape[1])


    local_path = '/media/gao/Gao106/NYUV2/data/nyu2_test/'
    bs = 2
    dst = nyuLoader(root=local_path, split='test')
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        image, depths, normal = data

        imgs = image.numpy()
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        imgs = imgs + 0.5

        normal = normal.numpy()
        normal = 0.5 * (normal + 1)
        normal = np.transpose(normal, [0, 2, 3, 1])

        # normal_mask = normal_mask.numpy()
        # normal_mask = np.repeat(normal_mask[:, :, :, np.newaxis], 3, axis=3)

        depths = depths.numpy()
        depths = np.transpose(depths, [0, 2, 3, 1])
        depths = np.repeat(depths, 3, axis=3)

        # raw_depth_mask = raw_depth_mask.numpy()
        # raw_depth_mask = np.repeat(raw_depth_mask[:, :, :, np.newaxis], 3, axis=3)


        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            # print(im_name[j])
            axarr[j][0].imshow(depths[j])
            # axarr[j][2].imshow(normal[j])
            # axarr[j][2].imshow(normal_mask[j])
            axarr[j][1].imshow(depths[j])
            # axarr[j][4].imshow(raw_depth_mask[j])

        plt.show()
        plt.close()
