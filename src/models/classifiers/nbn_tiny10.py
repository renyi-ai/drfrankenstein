'''LeNet in PyTorch.'''
import torch.nn as nn

from src.models.classifiers.general import GeneralNet


class NbnTiny10(GeneralNet):

    def __init__(self, n_classes, n_channels_in=1, seed=None, model_path=None):
        super(NbnTiny10, self).__init__(n_classes, n_channels_in, seed=seed, model_path=model_path)

        self.conv1 = nn.Conv2d(in_channels=n_channels_in, out_channels=16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.relu8 = nn.ReLU()

        self.fc = nn.Linear(in_features=64, out_features=n_classes)

        self.to_transform = [
            ('conv1', 'conv1.bias'),
            ('conv2', 'conv2.bias'),
            ('conv3', 'conv3.bias'),
            ('conv4', 'conv4.bias'),
            ('conv5', 'conv5.bias'),
            ('conv6', 'conv6.bias'),
            ('conv7', 'conv7.bias'),
            ('conv8', 'conv8.bias'),
        ]

        self._finish_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.relu8(x)

        x = x.mean(dim=(-2, -1))

        x = self.fc(x)
        return x
