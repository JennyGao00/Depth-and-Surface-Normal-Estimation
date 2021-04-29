import json
from loader.MT_loader import mtLoader
from loader.SC_loader import scLoader
from loader.NYU_loader import nyuLoader
from loader.Taskonomy_loader import TaskLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'matterport':mtLoader,
        'scannet':scLoader,
        'nyuv2': nyuLoader,
        'taskonomy': TaskLoader
    }[name]


def get_data_path(name, config_file='config.json'):
    """get_data_path

    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name]['data_path']