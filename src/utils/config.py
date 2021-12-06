import ast
import configparser as cp
import os
import types


def is_debugging():
    try:
        import pydevd
        DEBUGGING = True
    except ImportError:
        DEBUGGING = False
    if DEBUGGING:
        print("DEBUGGING")
    return DEBUGGING


def get_cuda_device():
    import torch
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cpu':
        print('WARNING: Could not set cuda, models will run on cpu')
    device = torch.device(device_name)
    return device


# Are we in debugging mode or not
debug = is_debugging()

# Should we put neural networks into lucid mode or not. It affects
# the activation functions: relu or redirected relu
lucid = False

# Torch device: cuda or cpu
device = get_cuda_device()


def get_config(custom_ini_file):
    file_conf = get_fileconf(custom_ini_file)
    conf = {}

    # Convert dictionaries to namespaces
    for section_name in file_conf.sections():
        d = {}
        for (key, val) in file_conf.items(section_name):
            d[key] = ast.literal_eval(val)

        item = types.SimpleNamespace(**d)
        conf[section_name] = item
    x = types.SimpleNamespace(**conf)
    return x


def get_fileconf(custom_ini_file):
    file_conf = cp.ConfigParser()
    root = os.path.split(custom_ini_file)[0]

    # Read in configs in hierarchy
    file_conf.read(os.path.join(root, 'default.env'))
    file_conf.read(os.path.join(root, 'default.ini'))
    file_conf.read(custom_ini_file)
    return file_conf
