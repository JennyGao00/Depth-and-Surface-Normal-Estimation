##########################################
# Train with RGB+D
# Note: 1. feature fusion at multiscale levels, different loss function
#       2. confidence map refined
#       3. Joint model under arch_N structure
#       4. load pretrain model from RGB_IN_l1 for encoder
# Last update on Oct.23, 2018, Jin Zeng
##########################################

import sys, os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import sklearn.preprocessing as sk
from os.path import join as pjoin
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model, get_lossfun
from loader import get_data_path, get_loader
from pre_trained import get_premodel
# from models.loss import cross_cosine, total_loss
from utils import norm_tf, get_fconv_premodel
from models.eval import evaluateError, addErrors, averageErrors
from models.depth_VNL import ModelLoss
from utils import merge_into_row_with_gt, save_image


def train(args):
    writer = SummaryWriter(comment=args.writer)

    # data loader setting, train and evaluat fconvion
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, split='train', img_size=(args.img_H, args.img_W), img_norm=args.img_norm)
    v_loader = data_loader(data_path, split='test', img_size=(args.img_H, args.img_W), img_norm=args.img_norm)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    evalloader = data.DataLoader(v_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Finish Loader Setup")

    # Setup Model and load pretrained model
    model_normal = args.arch_N
    model_N = get_model(model_normal, True)  # concat and output
    model_N = torch.nn.DataParallel(model_N, device_ids=range(torch.cuda.device_count()))
    model_N.cuda()
    state = torch.load(args.PATH)
    # model_N.load_state_dict(state)
    print("Finish model setup")

    # Setup depthmodel
    depth_model_name = args.arch_D
    DepthModel = get_model(depth_model_name, True)  # concat and output
    DepthModel = torch.nn.DataParallel(DepthModel, device_ids=range(torch.cuda.device_count()))
    DepthModel.cuda()
    # DepthModel.load_state_dict(torch.load(args.PATH))
    print("Finish DepthModel setup")

    # optimizers and lr-decay setting

    optimizer_N = torch.optim.RMSprop(model_N.parameters(), lr=args.l_rate)
    scheduler_N = torch.optim.lr_scheduler.MultiStepLR(optimizer_N, milestones=[10, 20, 40, 60], gamma=0.2)

    # optimizer_P = torch.optim.RMSprop(model_point.parameters(), lr=args.l_rate)
    # scheduler_P = torch.optim.lr_scheduler.MultiStepLR(optimizer_P, milestones=[10, 20, 30, 40], gamma=0.5)

    optimizer_D = torch.optim.RMSprop(DepthModel.parameters(), lr=0.1 * args.l_rate)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[10, 20, 40, 60], gamma=0.2)#second trial

    best_loss = 1
    n_iter_t, n_iter_v = 0, 0
    cos = nn.CosineSimilarity(dim=1, eps=0)
    total_iter_t = len(t_loader)
    loss_func = ModelLoss(fx=args.fx, fy=args.fy, img_size=(args.img_H, args.img_W))

    if not os.path.exists(args.model_savepath):
        os.makedirs(args.model_savepath)
    # forward and backward
    for epoch in range(args.n_epoch):
        model_N.train()
        DepthModel.train()

        # training
        for i, (images, depths, normal) in enumerate(trainloader):
            n_iter_t += 1
            images = Variable(images.contiguous().cuda())
            normal = Variable(normal.contiguous().cuda())
            depths = Variable(depths.contiguous().cuda())

            optimizer_N.zero_grad()
            optimizer_D.zero_grad()

            pca_normal, pointnet_feature= DepthModel(depths)  #
            coarse_normal, refined_normal = model_N(images, pca_normal, pointnet_feature)

            # result_img = merge_into_row_with_gt(images, depths, normal, coarse_normal, refined_normal)
            # save_image(result_img, args.model_savepath+'/00result'+str(i)+'.png')

            # loss_depth = loss_func.criterion(output_depth, depths)
            # loss_depth = torch.log(torch.abs(output_depth - depthes) + 0.5).mean()
            # loss_depth = total_loss(output_depth, depthes)

            # loss_pca_normal = torch.min(torch.sqrt(((normal.reshape(-1, 3) - pca_normal.reshape(-1, 3)) ** 2).sum(-1)),
            #   torch.sqrt(((normal.reshape(-1, 3) + pca_normal.reshape(-1, 3)) ** 2).sum(-1))).mean()
            loss_cN = torch.abs(1 - cos(coarse_normal, normal)).mean()
            loss_rN = torch.abs(1 - cos(refined_normal, normal)).mean()

            loss = loss_rN + loss_cN
            loss.backward()

            optimizer_N.step()
            scheduler_N.step()
            optimizer_D.step()
            scheduler_D.step()

            if (i+1) % 100== 0:
                # print("Train: Epoch [%d/%d], Iter [%d/%d], depth loss: %.4f" % (
                #       epoch + 1, args.n_epoch, i, total_iter_t, loss.data.item()))
                print("Train: Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, "
                        "coarse normal: %.4f, refine normal: %.4f" % (
                    epoch + 1, args.n_epoch, i+1, total_iter_t, loss.data.item(),
                    loss_cN.data.item(), loss_rN.data.item()))

            if (i + 1) % 2000 == 0:
                writer.add_scalar('loss/trainloss', loss.data.item(), n_iter_t)
                writer.add_images('RGB', images.detach().cpu().numpy() + 0.5, n_iter_t)
                writer.add_images('Depth_GT',
                                  np.repeat(((depths - torch.min(depths)) / (torch.max(depths) - torch.min(depths))).detach().cpu().numpy(),
                                            3, axis=1), n_iter_t)
                writer.add_images('Normal_GT', 0.5 * (normal.detach().cpu().numpy() + 1), n_iter_t)
                # writer.add_images('Pred_Depth',
                #                   np.repeat(((output_depth - torch.min(output_depth)) / (torch.max(output_depth) - torch.min(output_depth))).detach().cpu().numpy(),
                #                             3, axis=1), n_iter_t)
                writer.add_images('Coarse_Normal', 0.5 * (coarse_normal.detach().cpu().numpy() + 1), n_iter_t)
                writer.add_images('Refined_Normal', 0.5 * (refined_normal.detach().cpu().numpy() + 1), n_iter_t)

                # output_nd = norm_tf(output_depth)
                # outputs_cn = norm_tf(coarse_normal)
                # output_rd = norm_tf(refined_normal)

        model_N.eval()
        DepthModel.eval()
        mean_loss, sum_loss = 0, 0
        evalcount = 0
        # errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
        #       'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0,
        #       'mean': 0, 'median': 0, '11.25': 0, '22.5': 0, '30': 0}

        # evaluation
        with torch.no_grad():
            for i_val, (images_val, depths_val, normals_val) in tqdm(enumerate(evalloader)):
                n_iter_v += 1
                images_val = Variable(images_val.contiguous().cuda())
                normals_val = Variable(normals_val.contiguous().cuda())
                depths_val = Variable(depths_val.contiguous().cuda())

                pca_normal, pointnet_feature = DepthModel(depths_val)
                coarse_normal_val, refined_normal_val = model_N(images_val, pca_normal, pointnet_feature)

                # loss_depth = torch.log(torch.abs(output_depth_val - depthes_val) + 0.5).mean()
                # loss_depth = loss_func.criterion(output_depth_val, depths_val)
                # loss_pca_normal = torch.min(torch.sqrt(((normals_val - pca_normal_val) ** 2).sum(-1)),
                #                             torch.sqrt(((normals_val + pca_normal_val) ** 2).sum(-1))).mean()
                loss_cN = torch.abs(1 - cos(coarse_normal_val, normals_val)).mean()
                loss_rN = torch.abs(1 - cos(refined_normal_val, normals_val)).mean()

                loss = loss_cN + loss_rN

                if np.isnan(loss.detach().cpu().numpy()) | np.isinf(loss.detach().cpu().numpy()):
                    sum_loss += 0
                else:
                    sum_loss += loss.data.item()
                    evalcount += 1

                if (i_val+1) % 100==0:
                    # errors
                    errors = evaluateError(refined_normal_val, normals_val)
                    # errorSum = addErrors(errorSum, errors, args.batch_size)
                    # averageError = averageErrors(errorSum, args.batch_size)

                    print('metrics:')
                    print(errors)

                if (i_val+1) % 1000 == 0:
                    # print("Epoch [%d/%d] Evaluation Loss: %.4f" % (epoch+1, args.n_epoch, loss))
                    writer.add_scalar('loss/evalloss', loss.data.item(), n_iter_v)
                    writer.add_images('Eval_RGB', images_val + 0.5, n_iter_v)
                    writer.add_images('Eval_Depth_GT',
                                      np.repeat(
                                          ((depths_val - torch.min(depths_val)) / (torch.max(depths_val) - torch.min(depths_val))).detach().cpu().numpy(),
                                          3, axis=1), n_iter_v)
                    writer.add_images('Eval_Normal_GT', 0.5 * (normals_val.detach().cpu().numpy() + 1), n_iter_v)
                    # writer.add_images('Eval_Pred_Depth',
                    #                   np.repeat(((output_depth_val - torch.min(output_depth_val)) / (
                    #                               torch.max(output_depth_val) - torch.min(output_depth_val))).detach().cpu().numpy(),
                    #                             3, axis=1), n_iter_v)
                    writer.add_images('Eval_Coarse_Normal', 0.5 * (coarse_normal_val.detach().cpu().numpy() + 1), n_iter_v)
                    writer.add_images('Eval_Refined_Normal', 0.5 * (refined_normal_val.detach().cpu().numpy() + 1), n_iter_v)

            mean_loss = sum_loss / evalcount
            print("Epoch [%d/%d] Evaluation Mean Loss: %.4f" % (epoch + 1, args.n_epoch, mean_loss))
            writer.add_scalar('loss/evalloss_mean', mean_loss, epoch)

        # if mean_loss < best_loss:
        #     best_loss = mean_loss
        #     state = {'epoch': epoch + 1,
        #                  'model_N_state': model_N.state_dict(),
        #                  'optimizer_N_state': optimizer_N.state_dict(),
        #                  'DepthModel_state': DepthModel.state_dict(),
        #                  'optimizer_depth_state': optimizer_D.state_dict(), }
        #     torch.save(state, pjoin(args.model_savepath,
        #                                 "{}_{}_{}_best.pkl".format(args.arch_N, args.dataset, args.loss)))

    print('Finish training for dataset %s trial %s' % (args.dataset, args.model_num))
    writer.export_scalars_to_json("./{}_{}_{}.json".format(args.arch_N, args.dataset, args.loss))
    writer.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch_depth', nargs='?', type=str, default='vgg_16',
                        help='Architecture for depth to use [\'vgg_16, vgg_16_in etc\']')
    parser.add_argument('--arch_normal', nargs='?', type=str, default='unet_3',
                        help='Architecture for normal to use [\'unet_3, unet_3_mask, unet_3_mask_in etc\']')
    parser.add_argument('--arch_N', nargs='?', type=str, default='NormalModel',
                        help='Architecture for Fusion to use [\'fconv, fconv_in, fconv_ms etc\']')
    parser.add_argument('--arch_D', nargs='?', type=str, default='DepthModel',
                        help='Architecture for confidence map to use [\'mask, map_conv etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='taskonomy',
                        help='Dataset to use [\'nyuv2, matterport, scannet, taskonomy, etc\']')
    parser.add_argument('--img_H', nargs='?', type=int, default=240,
                        help='Height of the output image')  # default: nyu(240, 320); taskonomy(256, 256)
    parser.add_argument('--img_W', nargs='?', type=int, default=320,
                        help='Width of the output image')
    parser.add_argument('--fx', nargs='?', type=float, default=519.0,
                        help='focal_x')  
    parser.add_argument('--fy', nargs='?', type=float, default=519.0,
                        help='focal_y')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true',
                        help='Enable input image scales normalization [0, 1] | True by default')
    # parser.add_argument('--no-img_norm', dest='img_norm', action='store_false',
    #                     help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=30,
                        help='number of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0001,
                        help='Learning Rate')
    parser.add_argument('--PATH', nargs='?', type=str, default='./checkpoint/NormalModel_nyuv2_l1_best.pkl',
                        help='focal_y')

    parser.add_argument('--tfboard', dest='tfboard', action='store_true',
                        help='Enable visualization(s) on tfboard | False by default')
    # parser.add_argument('--no-tfboard', dest='tfboard', action='store_false',
    #                     help='Disable visualization(s) on tfboard | False by default')
    parser.set_defaults(tfboard=False)

    parser.add_argument('--state_name', nargs='?', type=str, default='vgg_16',
                        help='Path to the saved state dict, vgg_16, vgg_16_mp, vgg_16_mp_in')

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='previous saved model to restart from |False by default')
    parser.add_argument('--noresume', dest='resume', action='store_false',
                        help='donot use previous saved model to restart from | False by default')

    parser.add_argument('--resume_model_path', nargs='?', type=str, default='',
                        help='model path for the resume model')

    parser.set_defaults(resume=False)

    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint',
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_num', nargs='?', type=str, default='1',
                        help='Checkpoint index [\'1,2,3, etc\']')

    parser.add_argument('--depth_loss', nargs='?', type=str, default='VNL',
                        help='Loss type: VNL, l1')
    parser.add_argument('--loss', nargs='?', type=str, default='l1',
                        help='Loss type: cosine, sine, l1')

    parser.add_argument('--writer', nargs='?', type=str, default='geolearn',
                        help='writer comment: geolearn')
    parser.add_argument('--num_workers', nargs='?', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    train(args)
