##################################
# load taskonomy dataset
# JY Gao, 20210422
##################################

import os
from os.path import join as pjoin
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt
from loader.loader_utils import png_reader_32bit, png_reader_uint8
from torch.utils import data
from PIL import Image


class TaskLoader(data.Dataset):
    """Data loader for the taskonomy dataset.

    """

    def __init__(self, root, split, img_size=(256, 256), img_norm=True, mode=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.img_norm = img_norm
        # self.files = collections.defaultdict(list)
        self.files = []
        self.img_size = img_size if isinstance(img_size, tuple) \
            else (img_size, img_size)
        self.mode = mode

        # for split in ['train', 'test']:
        #     path = pjoin(self.root, 'taskonomy_' + split + '_list.txt')
        #     file_list = tuple(open(path, 'r'))
        #     file_list = [id_.rstrip() for id_ in file_list]
        #     self.files[split] = file_list

        path = pjoin(self.root, 'taskonomy_' + split + '_list.txt')
        file_list = tuple(open(path, 'r'))
        if split == 'train':
            file_list = [id_.rstrip() for id_ in file_list]
        if split == 'test':
            file_list = [id_.rstrip() for id_ in file_list]
        # self.files[split] = file_list
        self.files = file_list

    def __len__(self):
        # return len(self.files[self.split])
        return len(self.files)

    def __getitem__(self, index):
        # im_name_base = self.files[self.split][index]
        im_name_base = self.files[index]

        # raw_depth
        raw_depth_path = pjoin(self.root, str(im_name_base))
        raw_depth = png_reader_32bit(raw_depth_path, self.img_size)
        raw_depth[raw_depth == 65535] = 0.001
        raw_depth = raw_depth.astype(float)
        raw_depth = raw_depth / 10000
        raw_depth = raw_depth[np.newaxis, :, :]
        raw_depth = torch.from_numpy(raw_depth).float()

        # raw_depth_mask
        # raw_depth_mask = (raw_depth > 0.0001).astype(float)
        # raw_depth_mask = torch.from_numpy(raw_depth_mask).float()

        # image
        rgb_path = raw_depth_path.replace('depth_zbuffer', 'rgb')
        img = png_reader_uint8(rgb_path, self.img_size)
        img = img.astype(float)
        # img    = img / 255
        img = (img - 128) / 255
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        # normal
        normal_path = raw_depth_path.replace('depth_zbuffer', 'normal')
        normal = png_reader_uint8(normal_path, self.img_size)
        normal = normal.astype(float)
        normal = normal / 255
        normal = normal.transpose(2, 0, 1)
        normal = 2 * normal - 1
        normal = torch.from_numpy(normal).float()

        return img, raw_depth, normal





if __name__ == '__main__':
    # Config your local data path

    # depth_path = '/home/gao/depth.png'
    #
    # pred_depth = Image.open(depth_path)
    # depth = np.array(pred_depth)
    # depth = depth[:, :, 0]
    # depth = (depth - 128) / 255
    # depth = depth.astype(np.float32)
    # depth = depth.reshape(1, 1, depth.shape[0], depth.shape[1])


    local_path = '/media/gao/Gao106/taskonomy/'
    bs = 2
    dst = TaskLoader(root=local_path, split='train')
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


        f, axarr = plt.subplots(bs, 3)
        for j in range(bs):
            # print(im_name[j])
            axarr[j][0].imshow(imgs[j])
            # axarr[j][2].imshow(normal[j])
            # axarr[j][2].imshow(normal_mask[j])
            axarr[j][1].imshow(depths[j])
            axarr[j][2].imshow(normal[j])

        plt.show()
        plt.close()
