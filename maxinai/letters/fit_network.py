"""
Created on Dec 11, 2017

Train network model width "scikit-learn" framework

@author: Levan Tsinadze
"""

import torch.nn.functional as F
from skorch.net import NeuralNetClassifier
from torch import nn
from torch import optim
from utils.config import pytorch_config as pyconf

from maxinai.letters import training_flags as config
from maxinai.letters.dataset_utils import prepare_training
from maxinai.letters.network_model import choose_model


def test(test_loader, model, flags):
    """Test network
      Args:
        test_loader - test data loader
        model - network model
        flags - configuration parameters
    """

    model.eval()
    test_loss = 0
    correct = 0
    for (data, target) in test_loader:
        (data, target) = pyconf.attach_cuda(flags, (data, target))
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # Test and calculate loss and accuracy
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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

    model = choose_model(flags)
    pyconf.attach_cuda(flags, model)
    _attach_cpu(flags, model)

    (train_loader, val_loader, test_loader) = prepare_training(flags)
    net = NeuralNetClassifier(model,
                              criterion=nn.NLLLoss,
                              optimizer=optim.SGD,
                              optimizer__lr=flags.lr,
                              optimizer__momentum=flags.momentum,
                              iterator_train=train_loader,
                              iterator_valid=val_loader)
    print('End=attachment')
    net.fit()
    test(test_loader, model, flags)


if __name__ == '__main__':
    """Configure and train network"""

    flags = config.configure()
    run_training(flags)
