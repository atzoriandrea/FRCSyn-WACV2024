import torchvision.transforms as transforms
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.rand_augment import RandAugment


normalize_moco = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])




aug_h_flip = [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]

aug_rand_4_16 = [
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=4, magnitude=16),
    transforms.ToTensor(),
    normalize,
]
