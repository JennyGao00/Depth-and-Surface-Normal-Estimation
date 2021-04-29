import torchvision.models as models
from models.DepthModel import *
from models.NormalModel import *
from models.PointModel import *
from models.loss import *


def get_model(name, track_running_static):
    model = _get_model_instance(name)

    if name == 'NormalModel':
        model = model(input_channel=3, output_channel=3, track_running_static = track_running_static)
    elif name == 'DepthModel':
        model = model(input_channel=3, output_channel=1, track_running_static = track_running_static)
    return model

def _get_model_instance(name):
    try:
        return {
            'NormalModel': NormalModel,
            'DepthModel': DepthModel,
        }[name]
    except:
        print('Model {} not available'.format(name))

def get_lossfun(name, output_depth, depthes, pca_normal, coarse_normal, refined_normal, normal, train=True):
    lossfun = _get_loss_instance(name)

    #resize label, mask to input size
    if (input.size(2) != label.size(1)):
        if name == 'l1_sm':
            step = mask.size(1)/input.size(2)
            mask = mask[:, 0::step, :]
            mask = mask[:, :, 0::step]
        else:
            step = label.size(1)/input.size(2)
            label = label[:, 0::step, :, :]
            label = label[:, :, 0::step, :]
            mask = mask[:, 0::step, :]
            mask = mask[:, :, 0::step]

    loss, df = lossfun(input, label, mask, train)
    
    return loss, df

def _get_loss_instance(name):
    try:
        return {
            'cosine': cross_cosine,
            'sine': sin_cosine,
            'l1': l1norm,
            'l1gra': l1granorm,
            'l1_normgrad': l1_normgrad,
            'l1_sm': l1_sm,
            'energy': energy,
            'gradmap': gradmap,
        }[name]
    except:
        print('loss function {} not available'.format(name))
