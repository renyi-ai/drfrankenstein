'''LeNet in PxTorch.'''
import torch.nn as nn

from src.models.classifiers.general import GeneralNet


# class LeNet(GeneralNet):

#     def __init__(self, n_classes, seed=None, model_path=None):
#         super(LeNet, self).__init__(n_classes, seed=seed, model_path=model_path)

#         self.features = nn.Sequential(            
#             nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
#             nn.Tanh()
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=120, out_features=84),
#             nn.Tanh(),
#             nn.Linear(in_features=84, out_features=n_classes),
#         )

#         self._finish_init()

class LeNet(GeneralNet):
    def __init__(self, n_classes, n_channels_in=1, **kwargs):
        super(LeNet, self).__init__(n_classes, n_channels_in, **kwargs)
        self.conv1 = nn.Conv2d(n_channels_in, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, n_classes)

        self.to_transform = [('conv1', 'conv1.bias'), ('conv2', 'conv2.bias')]

        self._finish_init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)  # x.mean(dim=(-2, -1))
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x
