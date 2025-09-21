from .base import BaseDatasetInterface, BasicTorchDataset, BasicTorchWithPathsDataset
from .celeba import (
    CelebADatasetInterface,
    CelebAHQDatasetInterface,
    CelebAHQTorchDataset,
    CelebATorchDataset,
)
from .cub import CUBDatasetInterface
from .flowers import FlowersDatasetInterface
from .image_folder import ImageFolderDatasetInterface
from .imagenet import ImageNetDatasetInterface
from .mnist import MNISTDatasetInterface, MNISTTorchDataset
from .nabirds import NABirdsDatasetInterface
