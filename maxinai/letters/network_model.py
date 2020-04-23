"""
Created on Apr 23, 2020

Network model for handwritten character recognition

@author: Levan Tsinadze
"""

import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

logger = logging.getLogger(__name__)


class Flatten(nn.Module):
    """Flatten layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, 1)


class LetterNetClassic(nn.Module):
    """Network model without flatten layer
     for character recognition"""

    def __init__(self, input_channels=1):
        super(LetterNetClassic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        result = F.log_softmax(x, dim=1)

        return result


class ConvFeatures(nn.Module):
    """Convolutional features extractor"""

    def __init__(self, input_channels=1):
        super(ConvFeatures, self).__init__()
        self.conv_part = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(input_channels, 32, 3)),
                                                    ('bn1', nn.BatchNorm2d(32)),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('mxpl1', nn.MaxPool2d(2, 2)),
                                                    ('conv2', nn.Conv2d(32, 64, kernel_size=3)),
                                                    ('bn2', nn.BatchNorm2d(64)),
                                                    ('relu2', nn.ReLU(inplace=True)),
                                                    ('mxol2', nn.MaxPool2d(2, 2)),
                                                    ('drop1', nn.Dropout2d(p=0.25))]))

    def forward(self, input_tensor):
        return self.conv_part(input_tensor)


class LinearFeatures(nn.Module):
    """Extracts features vector"""

    def __init__(self):
        super(LinearFeatures, self).__init__()
        self.fc_part = nn.Sequential(OrderedDict([('flatten', Flatten()),
                                                  ('bn2', nn.BatchNorm1d(1024)),
                                                  ('relu3', nn.ReLU(inplace=True)),
                                                  ('drop2', nn.Dropout(p=0.25))]))

    def forward(self, input_tensor):
        return self.fc_part(input_tensor)


class LetterNetSlim(nn.Module):
    """Full letters network implementation"""

    def __init__(self, input_channels=1, num_classes=10):
        super(LetterNetSlim, self).__init__()
        self.conv_part = ConvFeatures(input_channels=input_channels)
        self.fc_part = LinearFeatures()
        self.fc = nn.Linear(1024, num_classes)

    def get_features(self, x):
        """
        Gets features vector from input
        Args:
            x: input tensor

        Returns:
            logits: result logits
        """
        x = self.conv_part(x)
        x = self.fc_part(x)
        logits = self.fc(x)

        return logits

    def forward(self, x):
        logits = self.get_features(x)
        result = F.log_softmax(logits, dim=1)

        return result


class LetterNet(nn.Module):
    """Full double letters network implementation"""

    def __init__(self, input_channels=1, num_classes=10):
        super(LetterNet, self).__init__()
        self.conv_part = ConvFeatures(input_channels=input_channels)
        self.dub_part = nn.Sequential(OrderedDict([('conv3', nn.Conv2d(64, 128, kernel_size=3)),
                                                   ('bn3', nn.BatchNorm2d(128)),
                                                   ('relu3', nn.ReLU(inplace=True)),
                                                   ('mxol3', nn.MaxPool2d(2, 2)),
                                                   ('drop2', nn.Dropout2d(p=0.25))]))
        self.fc_part = LinearFeatures()
        self.fc = nn.Linear(1024, num_classes)

    def get_features(self, x):
        """Gets logits from input
          Args:
            x - input tensor
          Returns:
            logits - result vector
        """

        x = self.conv_part(x)
        x = self.dub_part(x)
        x = self.fc_part(x)
        logits = self.fc(x)

        return logits

    def forward(self, x):
        logits = self.get_features(x)
        result = F.log_softmax(logits, dim=1)

        return result


def choose_model(flags):
    """Choose appropriated network model
      Args:
        flags - configuration flags
      Returns:
        model - network model
    """

    model_name = flags.model_name
    logger.debug('model_name - ', model_name)
    if model_name == 'LetterNet':
        model = LetterNet(input_channels=flags.channels, num_classes=flags.num_classes)
    elif model_name == 'LetterNetClassic':
        model = LetterNetClassic(input_channels=flags.channels, num_classes=flags.num_classes)
    elif model_name == 'LetterNetSlim':
        model = LetterNetSlim(input_channels=flags.channels, num_classes=flags.num_classes)
    else:
        model = LetterNet(input_channels=flags.channels, num_classes=flags.num_classes)

    return model
