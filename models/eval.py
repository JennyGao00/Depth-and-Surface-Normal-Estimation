############################
# Evaluate estimated normal
# criterion include:
# mean, median, 11.25, 22.5, 30
# Jin Zeng, 20180821
############################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc as m
import math

def eval_normal(input, label):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)

    mean = torch.mean(loss_angle)
    median = torch.median(loss_angle)
    val_num = loss_angle.size(0)
    small = torch.sum(torch.lt(loss_angle, 11.25)).to(torch.float)/val_num
    mid = torch.sum(torch.lt(loss_angle, 22.5)).to(torch.float)/val_num
    large = torch.sum(torch.lt(loss_angle, 30)).to(torch.float)/val_num

    return mean.data.item(), median.data.item(), small.data.item(), mid.data.item(), large.data.item()

def eval_normal_pixel(input, label, mask):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    mask_t = mask.view(-1,1)
    mask_t = torch.squeeze(mask_t)

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)
    loss_angle = loss_angle[torch.nonzero(mask_t)]

    val_num = loss_angle.size(0) 

    if val_num>0:
        mean = torch.mean(loss_angle).data.item()
        median = torch.median(loss_angle).data.item()    
        small = (torch.sum(torch.lt(loss_angle, 11.25)).to(torch.float)/val_num).data.item()
        mid = (torch.sum(torch.lt(loss_angle, 22.5)).to(torch.float)/val_num).data.item()
        large = (torch.sum(torch.lt(loss_angle, 30)).to(torch.float)/val_num).data.item()
    else:
        mean=0
        median=0
        small=0
        mid=0
        large=0

    outputs_n = 0.5*(input_v+1)                
    outputs_n = outputs_n.view(-1, h, w, ch)# bs*h*w*3

    return outputs_n, val_num, mean, median, small, mid, large

def eval_normal_detail(input, label, mask):
    # bs = 1 for testing
    # input: bs*ch*h*w
    # label: bs*h*w*ch
    # mask: bs*h*w
    bz, ch, h, w = input.size()
    
    # normalization
    input = input.permute(0,2,3,1).contiguous().view(-1,ch)
    input_v = F.normalize(input,p=2)    
    label_v = label.contiguous().view(-1,ch)
    # label_v = F.normalize(label_v,p=2) 
    # input_v[torch.isnan(input_v)] = 0

    mask_t = mask.view(-1,1)
    mask_t = torch.squeeze(mask_t)

    loss = F.cosine_similarity(input_v, label_v)#compute inner product     
    loss[torch.ge(loss,1)] = 1
    loss[torch.le(loss,-1)] = -1  
    loss_angle = (180/np.pi)*torch.acos(loss)
    loss_angle = loss_angle[torch.nonzero(mask_t)] 

    return loss_angle

def eval_print(sum_mean, sum_median, sum_small, sum_mid, sum_large, sum_num, item='Pixel-Level'):
    allnum = sum(sum_num)
    if allnum == 0:
        print("Empty in %s pixels" % (item))
    else:
        pixel_mean = np.sum(np.array(sum_mean)*np.array(sum_num))/allnum   
        pixel_median = np.sum(np.array(sum_median)*np.array(sum_num))/allnum    
        pixel_small = np.sum(np.array(sum_small)*np.array(sum_num))/allnum   
        pixel_mid = np.sum(np.array(sum_mid)*np.array(sum_num))/allnum   
        pixel_large = np.sum(np.array(sum_large)*np.array(sum_num))/allnum   

        print("Evaluation %s Mean Loss: mean %.4f, median %.4f, 11.25 %.4f, 22.5 %.4f, 30 %.4f" % (item, 
        pixel_mean, pixel_median, pixel_small, pixel_mid, pixel_large))                         

def eval_mask_resize(segment_val, img_rows, img_cols):
    segment_val = np.squeeze(segment_val.data.cpu().numpy(), axis=0)#uint8 array
    segment_val = m.imresize(segment_val, (img_rows, img_cols))#only works for 8 bit image
    segment_val = segment_val>0
    segment_val = torch.from_numpy(segment_val.astype(np.uint8))
    segment_val = Variable(segment_val.contiguous().cuda())

    return segment_val


def lg10(x):
    return torch.div(torch.log(x), math.log(10))


def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z

def nValid(x):
    return torch.sum(torch.eq(x, x).float())


def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())


def getNanMask(x):
    return torch.ne(x, x)


def setNanToZero(input, target):
    nanMask = getNanMask(target)
    nValidElement = nValid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(refine_normal, normal):
    errors = {'mean': 0, 'median': 0, '11.25': 0, '22.5': 0, '30': 0}
    # errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
    #           'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0,
    #           'mean': 0, 'median': 0, '11.25': 0, '22.5': 0, '30': 0}

    # _output, _target, nanMask, nValidElement = setNanToZero(output, target)
    # _refinenormal, _normal, nanNMask, nNValidElement = setNanToZero(refine_normal, normal)
    bz, ch, h, w = normal.size()

    # if (nValidElement.data.cpu().numpy() > 0):
    #     diffMatrix = torch.abs(_output - _target)
    #
    #     errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement
    #
    #     errors['MAE'] = torch.sum(diffMatrix) / nValidElement
    #
    #     realMatrix = torch.div(diffMatrix, _target)
    #     realMatrix[nanMask] = 0
    #     errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement
    #
    #     LG10Matrix = torch.abs(lg10(_output) - lg10(_target))
    #     LG10Matrix[nanMask] = 0
    #     errors['LG10'] = torch.sum(LG10Matrix) / nValidElement
    #     yOverZ = torch.div(_output, _target)
    #     yOverZ = torch.abs(yOverZ)
    #     zOverY = torch.div(_target, _output)
    #     zOverY = torch.abs(zOverY)
    #
    #     maxRatio = maxOfTwo(yOverZ, zOverY)
    #
    #     errors['DELTA1'] = torch.sum(
    #         torch.le(maxRatio, 1.25).float()) / nValidElement
    #     errors['DELTA2'] = torch.sum(
    #         torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
    #     errors['DELTA3'] = torch.sum(
    #         torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement
    #
    #
    #
    #     errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
    #     errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
    #     errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
    #     errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
    #     errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
    #     errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
    #     errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())


    # normalization
    refine_normal = refine_normal.permute(0, 2, 3, 1).contiguous().view(-1, ch)
    # refine_normal = F.normalize(refine_normal, p=2)
    normal = normal.contiguous().view(-1, ch)
    # label_v = F.normalize(label_v,p=2)
    # input_v[torch.isnan(input_v)] = 0


    loss = F.cosine_similarity(refine_normal, normal)  # compute inner product
    loss[torch.ge(loss, 1)] = 1
    loss[torch.le(loss, -1)] = -1
    loss_angle = (180 / np.pi) * torch.acos(loss)

    mean = torch.mean(loss_angle)
    median = torch.median(loss_angle)
    val_num = loss_angle.size(0)
    small = torch.sum(torch.lt(loss_angle, 11.25)).to(torch.float) / val_num
    med = torch.sum(torch.lt(loss_angle, 22.5)).to(torch.float) / val_num
    large = torch.sum(torch.lt(loss_angle, 30)).to(torch.float) / val_num

    errors['mean'] = mean.data.item()
    errors['median'] = median.data.item()
    errors['11.25'] = small.data.item()
    errors['22.5'] = med.data.item()
    errors['30'] = large.data.item()

    return errors


def addErrors(errorSum, errors, batchSize):
    errorSum['MSE'] = errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['ABS_REL'] = errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
    errorSum['LG10'] = errorSum['LG10'] + errors['LG10'] * batchSize
    errorSum['MAE'] = errorSum['MAE'] + errors['MAE'] * batchSize

    errorSum['DELTA1'] = errorSum['DELTA1'] + errors['DELTA1'] * batchSize
    errorSum['DELTA2'] = errorSum['DELTA2'] + errors['DELTA2'] * batchSize
    errorSum['DELTA3'] = errorSum['DELTA3'] + errors['DELTA3'] * batchSize

    errorSum['mean'] = errorSum['mean'] + errors['mean'] * batchSize
    errorSum['median'] = errorSum['median'] + errors['median'] * batchSize
    errorSum['11.25'] = errorSum['11.25'] + errors['11.25'] * batchSize
    errorSum['22.5'] = errorSum['22.5'] + errors['22.5'] * batchSize
    errorSum['30'] = errorSum['30'] + errors['30'] * batchSize

    return errorSum


def averageErrors(errorSum, N):
    averageError = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                    'MAE': 0, 'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0,
                    'mean': 0, 'median': 0, '11.25': 0, '22.5': 0, '30': 0}

    averageError['MSE'] = errorSum['MSE'] / N
    averageError['ABS_REL'] = errorSum['ABS_REL'] / N
    averageError['LG10'] = errorSum['LG10'] / N
    averageError['MAE'] = errorSum['MAE'] / N

    averageError['DELTA1'] = errorSum['DELTA1'] / N
    averageError['DELTA2'] = errorSum['DELTA2'] / N
    averageError['DELTA3'] = errorSum['DELTA3'] / N

    averageError['mean'] = errorSum['mean'] / N
    averageError['median'] = errorSum['median'] / N

    averageError['11.25'] = errorSum['11.25'] / N
    averageError['22.5'] = errorSum['22.5'] / N
    averageError['30'] = errorSum['30'] / N

    return averageError
