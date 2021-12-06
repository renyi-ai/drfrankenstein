import os

import torch

from src.dataset import get_n_classes_and_channels

celeba_models = [
                    'imagenet', 'adam', 'adam_overfit', 'sgd', 'img-gn1', 'img-gn2', 'img-gn3',
                    'in1-gn1', 'in1-gn2', 'in1-gn3', 'in2-gn1', 'in2-gn2', 'in2-gn3',
                    'in3-gn1', 'in3-gn2', 'in3-gn3', 'img-gn1-bn', 'in1-gn1-bn', 'in2-gn1-bn',
                    'in3-gn1-bn'
                ] + [f"FRANK_in{i}-gn{i}" for i in range(20)]


def load_from_path(path):
    # Get target device
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    # Skeleton model
    model = get_skeleton_model(path)
    model.to(device)

    # Load weights
    if path in celeba_models:
        celeba_name = path
        base_url = 'https://no_url/'
        url = base_url + celeba_name + '.pt'
        state_dict = torch.hub.load_state_dict_from_url(url,
                                                        progress=True,
                                                        map_location=device)
    else:
        state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_info_from_path(path):
    if path in celeba_models:
        model_name = 'inceptionv1'
        n_classes = 40
        n_channels = 3
        data_name = 'celeba'
    else:
        path_split = path.split(os.sep)
        model_name = path_split[-4]
        data_name = path_split[-3]
        n_classes, n_channels = get_n_classes_and_channels(data_name)
    return model_name, data_name, n_classes, n_channels


def get_skeleton_model(path):
    ''' Reads in a model architecture with random init '''
    model_name, _, n_classes, n_channels = get_info_from_path(path)
    model = get_model(model_name, n_classes, n_channels)
    return model


def get_model(str_nn,
              n_classes,
              n_channels,
              seed=None,
              model_path=None,
              celeba_name=None):
    str_nn = str_nn.lower()

    if str_nn == 'lenet':
        from src.models import LeNet
        model = LeNet(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'tiny10':
        from src.models import Tiny10
        model = Tiny10(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn in ['nbn_tiny10', 'nbntiny10']:
        from src.models import NbnTiny10
        model = NbnTiny10(n_classes,
                          n_channels,
                          seed=seed,
                          model_path=model_path)
    elif str_nn == 'dense':
        from src.models import Dense
        model = Dense(n_classes, n_channels, seed=seed, model_path=model_path)
    elif str_nn == 'inceptionv1':
        from src.models import InceptionV1
        model = InceptionV1(n_classes, n_channels, celeba_name=celeba_name)
    elif str_nn[:8] == 'resnet20':
        parts = str_nn.split('_')
        width = 1
        if len(parts) > 1:
            width = int(parts[1][1:])
        from src.models import ResNet20
        model = ResNet20(n_classes,
                         n_channels,
                         seed=seed,
                         model_path=model_path,
                         width=width)
    else:
        raise ValueError('Network {} is not known.'.format(str_nn))

    return model
