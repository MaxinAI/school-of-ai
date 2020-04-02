"""
Created on Nov 30, 2017

Data preprocessing for training

@author: Levan Tsinadze
"""

import json
import logging

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import (datasets, transforms)
from utils.config import pytorch_config as pyconf
from utils.files import file_utils

from maxinai.letters.cnn_files import files as _files

logger = logging.getLogger(__name__)


def _transform_func():
    """Initializes transformation functions cascade"""
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def prepare_mnist(flags, **kwargs):
    """Prepares training data loaders
      Args:
        flags - configuration flags
      Returns:
        tuple of -
          train_loader - training data loader
          val_loader  - validation data loader
          test_loader - test data loader
    """

    train_loader = DataLoader(
        datasets.MNIST(_files.data_dir, train=True, download=True, transform=_transform_func()),
        batch_size=flags.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        datasets.MNIST(_files.data_dir, train=False, transform=_transform_func()),
        batch_size=flags.test_batch_size, shuffle=True, **kwargs)

    return (train_loader, None, test_loader)


def pil_loader_gr(path):
    # open path as file to avoid
    # ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def _save_labels(flags, class_idxs):
    """Serializes model labels in JSON file
      Atgs:
        flags - configuration parameters
        class_idxs - indices of classes
    """

    labels_file = flags.labels

    if labels_file is not None:
        model_labels = {str(class_name): class_idx for (class_idx, class_name) in class_idxs.iteritems()}
        logger.debug('model_labels - ', model_labels)
        file_utils.delete(labels_file)
        with open(labels_file, 'w') as fp:
            json.dump(model_labels, fp)


def read_labels(flags):
    """Reads labels
      Args:
        flags - configuration parameters
      Returns:
        model_labels - labels JSON dictionary
    """

    labels_file = flags.labels
    if labels_file is not None:
        with open(labels_file, 'r') as fp:
            model_labels = json.load(fp)
            logger.debug('model_labels - ', model_labels)
    else:
        model_labels = {}

    return model_labels


def prepare_geoletters(flags, **kwargs):
    """Prepares training data loaders
      Args:
        flags - configuration flags
      Returns:
        tuple of -
          train_loader - training data loader
          val_loader - validation data loader
          test_loader - test data loader
    """

    train_dataset = datasets.ImageFolder(flags.train_dir,
                                         loader=pil_loader_gr,
                                         transform=_transform_func())
    train_loader = DataLoader(
        train_dataset,
        batch_size=flags.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(
        datasets.ImageFolder(flags.val_dir, loader=pil_loader_gr, transform=_transform_func()),
        batch_size=flags.test_batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(
        datasets.ImageFolder(flags.test_dir, loader=pil_loader_gr, transform=_transform_func()),
        batch_size=flags.test_batch_size, shuffle=False, **kwargs)
    class_names = train_dataset.classes
    class_idxs = train_dataset.class_to_idx
    logger.debug('class_names - ', class_names, '\n', 'class_idxs - ', class_idxs)
    flags.num_classes = len(class_names)
    _save_labels(flags, class_idxs, class_names)

    return (train_loader, val_loader, test_loader)


def prepare_training(flags):
    """Prepares data for training
      Args:
        flags - configuration flags
      Returns:
        tuple of -
          train_loader - training data loader
          val_loader validation data loader
          test_loader - test data loader
    """

    torch.manual_seed(flags.seed)
    pyconf.cuda_seed(flags)
    kwargs = {'num_workers': flags.num_workers, 'pin_memory': True} if flags.cuda else {}
    if flags.geoletters:
        logger.debug('loading letters data')
        (train_loader, val_loader, test_loader) = prepare_geoletters(flags, **kwargs)
    else:
        logger.debug('loading MNIST data')
        (train_loader, val_loader, test_loader) = prepare_mnist(flags, **kwargs)

    return (train_loader, val_loader, test_loader)
