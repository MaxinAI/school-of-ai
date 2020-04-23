"""
Created on Nov 7, 2017

Trains network for character recognition

@author: Levan Tsinadze
"""

import logging

import torch
import torch.nn.functional as F
from maxinai.config import pytorch_config as pyconf
from torch import nn
from torch import optim

from maxinai.letters import training_flags as config
from maxinai.letters.dataset_utils import prepare_training
from maxinai.letters.network_model import choose_model

logger = logging.getLogger(__name__)


def validate_test(data_loader, model, flags, testing=False):
    """Test network
      test_loader - test data loader
      model - network model
      flags - configuration parameters
      testing - flag for testing
    """

    model.eval()
    val_loss = 0
    correct = 0
    val_message = 'Test' if testing else 'Validation'
    for (data, target) in data_loader:
        (data, target) = pyconf.attach_cuda(flags, (data, target))
        output = model(data)
        val_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= len(data_loader.dataset)
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_message, val_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def train(epoch, training_config):
    """Train network model
      Args:
        epoch - current epoch
        training_config - training configuration tuple
    """

    (train_loader, _, model, optimizer, flags) = training_config
    criterion = nn.CrossEntropyLoss() if flags.geoletters else nn.NLLLoss()
    model.train()
    batch_idx = 0
    for (data, target) in train_loader:
        (data, target) = pyconf.attach_cuda(flags, (data, target))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_idx += 1
        if batch_idx % flags.log_interval == 0:
            #f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.data[0]:.6f}'
            print(.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
            torch.save(model.state_dict(), flags.weights)

    def _attach_cpu(flags, model):
        """Attaches model on CPU if flag is on
          Args:
            flags - network configuration parameters
            model - network model
        """

        if flags.attach_cpu:
            model.cpu()

    def run_training(flags):
        """Initializes and trains network model
          Args:
            flags - network configuration parameters
        """

        (train_loader, val_loader, test_loader) = prepare_training(flags)
        logger.debug('Prepare training data')
        model = choose_model(flags)
        pyconf.attach_cuda(flags, model)
        _attach_cpu(flags, model)
        logger.debug('model_type and instance - ', type(model), model)

        logger.debug('End=attachment')
        optimizer = optim.SGD(model.parameters(), lr=flags.lr, momentum=flags.momentum)
        training_config = (train_loader, val_loader, model, optimizer, flags)
        testing = val_loader is None
        val_loader = test_loader if testing else val_loader
        for epoch in range(1, flags.epochs + 1):
            train(epoch, training_config)
            validate_test(val_loader, model, flags, testing=testing)
        # Run final test if test set loader is valid
        if not testing:
            validate_test(test_loader, model, flags, testing=True)

    if __name__ == '__main__':
        flags = config.configure()
        run_training(flags)
