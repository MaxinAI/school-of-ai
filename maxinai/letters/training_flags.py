"""
Created on Nov 7, 2017

Training configuration for character detection

@author: Levan Tsinadze
"""

import argparse

from torch import cuda

from maxinai.letters.cnn_files import files as _files

MODEL_LABELS = 'geoletters_labels.json'


def configure():
    """Configuration parameters
      Returns:
        flags - configuration parameters
    """

    parser = argparse.ArgumentParser(description='Network models for characters recognition')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval',
                        type=int,
                        default=10,
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model_name',
                        type=str,
                        default='LetterNet',
                        help='Network model name to use')
    parser.add_argument('--attach_cpu',
                        action='store_true',
                        default=False,
                        help='Attach model to CPU device')
    # Input configuration
    parser.add_argument('--channels',
                        type=int,
                        default=1,
                        help='Input image channels')
    parser.add_argument('--height',
                        type=int,
                        default=28,
                        help='Input image height')
    parser.add_argument('--width',
                        type=int,
                        default=28,
                        help='Input image width')
    parser.add_argument('--thumb_height',
                        type=int,
                        default=28,
                        help='Input image maximum height before processing')
    parser.add_argument('--thumb_width',
                        type=int,
                        default=28,
                        help='Input image maximum width before processing')
    parser.add_argument('--border',
                        type=int,
                        default=8,
                        help='Padding border of image before interface')
    parser.add_argument('--make_borders',
                        dest='make_borders',
                        action='store_true',
                        default=False,
                        help='Flag for make borders in image tensor.')
    parser.add_argument('--border_px',
                        type=int,
                        default=4,
                        help='Copy and make borders in image tensor')
    # Save training model
    parser.add_argument('--weights',
                        type=str,
                        default=_files.model_file('mnist_weights.pth.tar'),
                        help='Where to save trained weights')
    # Data set configuration
    parser.add_argument('--dataset_dir',
                        type=str,
                        default=_files.data_file('dataset'),
                        help='Path to folders of labeled images for pre-processing and training.')
    parser.add_argument('--train_dir',
                        type=str,
                        default=_files.data_file('training'),
                        help='Path to folders of labeled images for training.')
    parser.add_argument('--val_dir',
                        type=str,
                        default=_files.data_file('validation'),
                        help='Path to folders of labeled images for validation.')
    parser.add_argument('--val_precentage',
                        type=float,
                        default=0.2,
                        help='Percentage of validation data.')
    parser.add_argument('--test_dir',
                        type=str,
                        default=_files.data_file('test'),
                        help='Path to folders of labeled images for testing.')
    parser.add_argument('--test_precentage',
                        type=float,
                        default=0.2,
                        help='Percentage of test data.')
    parser.add_argument('--geoletters',
                        dest='geoletters',
                        action='store_true',
                        default=False,
                        help='Flag for training on Georgian letters data.')
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='Number of output classes')
    # System configuration
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of data loader workers')
    # Logging configuration
    parser.add_argument('--print_dataset',
                        dest='print_dataset',
                        action='store_true',
                        help='Prints data set file names and labels.')
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='Flag for debug mode')
    # Serve applications
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        help='Service host')
    parser.add_argument('--port',
                        type=int,
                        default=8080,
                        help='Service host')
    flags = parser.parse_args()
    flags.cuda = not flags.no_cuda and cuda.is_available()
    flags.weights = _files.model_file('geoletters_weights.pth.tar') \
        if flags.geoletters \
        else _files.model_file('mnist_weights.pth.tar')
    flags.border = 0 if flags.make_borders else flags.border
    flags.labels = _files.model_file(MODEL_LABELS) if flags.geoletters else None

    return flags
