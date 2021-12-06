'''LeNet in PxTorch.'''
import torch.nn as nn

from src.models.classifiers.general import GeneralNet


class Dense(GeneralNet):
    def __init__(self, n_classes, n_channels_in=1, **kwargs):
        super(Dense, self).__init__(n_classes, n_channels_in, **kwargs)

        self.fc1 = nn.Linear(n_channels_in * 32 * 32, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 300)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(300, 300)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(300, n_classes)

        self.to_transform = [('fc1', 'fc1.bias'),
                             ('fc2', 'fc2.bias'),
                             ('fc3', 'fc3.bias'),
                             ('fc4', 'fc4.bias')]

        self._finish_init()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
