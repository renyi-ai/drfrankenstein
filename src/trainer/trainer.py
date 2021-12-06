import os

import pkbar
import torch
from dotmap import DotMap
from packaging import version
from torch.utils.data import DataLoader

from src.utils import config


class Trainer:
    def __init__(self,
                 datasets,
                 model_trainer,
                 gradient_noise=None,
                 batch_size=32,
                 n_workers=4,
                 drop_last=True,
                 save_folder='snapshots',
                 lr_schedule=[(0.333, 0.1), (0.666, 0.1)],
                 lr_schedule_type='step_function'):

        self.datasets = datasets
        self.model_trainer = model_trainer
        self.gradient_noise = gradient_noise
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.drop_last = drop_last
        self.save_folder = save_folder
        self.lr_schedule = lr_schedule
        self.lr_schedule_index = 0
        self.lr_schedule_type = lr_schedule_type

        self.model = model_trainer.model
        self.data_loaders = self._create_data_loaders(self.datasets,
                                                      self.gradient_noise,
                                                      self.n_workers,
                                                      self.drop_last)

    @property
    def _pbar_length(self):
        data_loader = self.data_loaders.train
        return self._get_n_iter(data_loader)

    @property  # Should be cached_property on python 3.8
    def save_subfolder(self):
        data_name = str(self.datasets.train.__class__.__name__)
        in_folder = 'in' + str(self.model.seed)
        gn_folder = 'gn' + str(self.gradient_noise)
        initfolder = in_folder + '-' + gn_folder
        folder = os.path.join(self.save_folder, self.model.name, data_name,
                              initfolder)
        return folder

    def _get_n_iter(self, data_loader):
        return len(data_loader.dataset) // data_loader.batch_size

    def _get_pbar(self, n, i, n_epochs):
        return pkbar.Kbar(target=n,
                          epoch=i,
                          num_epochs=n_epochs,
                          width=8,
                          always_stateful=False)

    def _update_running_metrics(self, orig, new):
        for k, v in new.items():
            orig[k] += new[k]
        return orig

    def save(self, n_iter_ran):
        self.model_trainer.save(n_iter_ran, self.save_subfolder)

    def _append_and_save_log(self, epoch_data):
        self.model_trainer.append_and_save_log(epoch_data, self.save_subfolder)

    def _update_lr(self, progress):
        if self.lr_schedule_index >= len(self.lr_schedule):
            return
        next_schedule = self.lr_schedule[self.lr_schedule_index]
        progress_threshold = next_schedule[0]
        if progress >= progress_threshold:
            multiplier = next_schedule[1]
            for g in self.model_trainer.optimizer.param_groups:
                g['lr'] *= multiplier
            self.lr_schedule_index += 1

    def train(self, epochs=10, save_frequency=None, freeze_bn=False):

        self.model.to(config.device)

        # Turn on running mean and std for batch norm
        if freeze_bn:
            self.model.eval()
        else:
            self.model.train()

        n_iterations_ran = 0

        if save_frequency is not None:
            self.save(0)

        if self.lr_schedule_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.model_trainer.optimizer, factor=0.1)

        for epoch_i in range(epochs):

            # Metrics and progress bar
            epoch_data = {}
            pbar = self._get_pbar(self._pbar_length, epoch_i, epochs)
            if self.lr_schedule_type == 'step_function':
                self._update_lr(epoch_i / epochs)

            for phase in ['train', 'val']:

                # Init metrics and data load dataloader
                running_metrics = {'loss': 0., 'accuracy': 0.}
                data_loader = self.data_loaders[phase]
                n_iter = self._get_n_iter(data_loader)

                if phase == 'val':
                    self.model_trainer.model.eval()
                elif not freeze_bn:
                    self.model_trainer.model.train()
                for i, (inputs, labels) in enumerate(data_loader):

                    # Make a train step and retreive metrics
                    new_metrics = self.model_trainer.step(
                        inputs, labels, phase == 'train')
                    running_metrics = self._update_running_metrics(
                        running_metrics, new_metrics)
                    new_values = [(phase + '_' + name, value)
                                  for (name, value) in new_metrics.items()]
                    # Save model checkpoint
                    if phase == 'train':
                        pbar.update(i, values=new_values)
                        n_iterations_ran += 1
                        if save_frequency and n_iterations_ran % save_frequency == 0:
                            self.save(n_iterations_ran)

                # Save information about this epoch
                for name, value in running_metrics.items():
                    if name != 'loss':
                        value = value.detach().cpu().numpy()
                    epoch_data[phase + '_' + name] = value / n_iter

                if phase == 'val':
                    val_values = [(x, y) for (x, y) in epoch_data.items()
                                  if x.startswith('val_')]
                    pbar.add(1, values=val_values)
                    if self.lr_schedule_type == 'reduce_on_plateau':
                        scheduler.step(epoch_data['val_loss'])

            # Save training log
            self._append_and_save_log(epoch_data)

    def _create_data_loaders(self, datasets, gradient_noise, n_workers,
                             drop_last):

        if self.gradient_noise is not None:
            torch.manual_seed(gradient_noise)

        high_end_version = version.parse(
            torch.__version__) >= version.parse("1.7.0")

        common_settings = dict(batch_size=self.batch_size,
                               num_workers=n_workers,
                               pin_memory=True,
                               drop_last=drop_last)
        if high_end_version and n_workers > 0:
            common_settings['prefetch_factor'] = 10

        train = DataLoader(datasets.train, shuffle=True, **common_settings)
        val = DataLoader(datasets.val, **common_settings)

        data_loaders = DotMap({'train': train, 'val': val})
        return data_loaders
