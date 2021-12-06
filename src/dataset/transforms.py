import warnings
warnings.filterwarnings("ignore", category=UserWarning) # suppress deprecation warning coming from torchvision
from torchvision import transforms as T

transformer = {

    'cifar10': [
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
        ],
        [
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010))
        ]
    ],

    'cifar100': [
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, 4),
            T.ToTensor(),
            T.Normalize(mean=(0.5071, 0.4867, 0.4408),
                        std=(0.2675, 0.2565, 0.2761))
        ],
        [
            T.ToTensor(),
            T.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=(0.2675, 0.2565, 0.2761))
        ]
    ],

    'mnist': [
        [
            T.Resize((32, 32)),
            T.Pad(4, padding_mode='edge'),
            T.RandomAffine(5, scale=(0.9, 1.1), shear=5, fillcolor=0),
            T.CenterCrop(32),
            T.ToTensor(),
            T.Normalize(mean=[0.1307], std=[0.3081])
        ],
        [
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize(mean=[0.1307], std=[0.3081])
        ]
    ],

    'celeba': [
        [
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.Pad(10, padding_mode='edge'),
            T.RandomRotation(10),
            T.CenterCrop(224),
            T.ToTensor(),
            lambda x: x * 255 - 117
        ],
        [
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            lambda x: x * 255 - 117
        ]
    ]
}
