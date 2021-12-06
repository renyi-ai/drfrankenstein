import torch
import torch.utils.data as data_utils

from src.dataset import get_datasets
from src.utils import config


def _get_data_loader(dataset_name,
                     dataset_type,
                     batch_size=256,
                     seed=None):
    ''' Get pytorch dataloader given the dataset_name and the type
        (train or val)'''

    if seed is not None:
        torch.manual_seed(seed)

    dataset = get_datasets(dataset_name)[dataset_type]
    if config.debug:
        debug_size = min(3 * batch_size + 1, len(dataset))
        dataset = data_utils.Subset(dataset, torch.arange(debug_size))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              num_workers=0 if config.debug else 4,
                                              pin_memory=True,
                                              drop_last=False)
    return data_loader
