import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CelebAGenerator(Dataset):
    """ Generator for lucid InceptionV1 GoogLeNet
        Args:
            csv (str):
                The path to the csv, placed in the same folder
                which holds the images.
            target_shape (tuple, optional):
                The input shape of the neural network.
                Defaults to (224, 224, 3).
            batch_size (int, optional):
                (Maximum) batch size. Defaults to 32.
            shuffle (bool, optional):
                Either to shuffle the data after each epoch and at the
                beginning of the training. Defaults to False.
            augment_func (func, optional):
                Augmentation function. Defaults to None.
            preprocess_func (str, optional):
                Image preprocessing function, applied on RGB uint8 input.
                Can be either 'imagenet' or 'robi'.
                Defaults to 'imagenet'.
    """

    def __init__(self,
                 csv,
                 transform=None,
                 target_shape=(224, 224, 3)):

        # Read in csv and define root directory
        self.df = pd.read_csv(csv)
        self.root = os.path.split(csv)[0]
        self.classes = list(self.df.columns.drop('img'))

        # transforms
        self.transform = transform

        # Convert relative paths to absolute
        self.df.img = self.df.img.apply(lambda x: os.path.join(self.root, x))

        self.target_shape = target_shape

    @classmethod
    def from_conf(cls, conf, is_train):
        if is_train:
            csv = conf.path.train_df
            transform = rich_augment if conf.train.rich_augment else augment_trans
        else:
            csv = conf.path.test_df
            transform = preprocess_trans

        return cls(csv=csv,
                   transform=transform,
                   target_shape=conf.train.image_shape)

    def __len__(self):
        ''' The length of the generator. Df size / batch_size '''
        return len(self.df)

    def __getitem__(self, i):
        ''' Get i^th item of the generator as gen[i] '''
        row = self.df.iloc[int(i)]

        # Input
        x = cv2.imread(row.img)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            x = self.transform(x)

        # Output
        y = row.drop('img').astype(np.float32).values
        y = (y + 1) / 2.  # [-1,1] data to [0,1]

        return x, y
