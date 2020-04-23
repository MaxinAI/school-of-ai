"""
Created on Nov 7, 2017

Training configuration for character detection

@author: Levan Tsinadze
"""

import argparse

from path_utils import data_path

MODEL_LABELS = 'geoletters_labels.json'


def configure():
    """Configuration parameters
      Returns:
        flags - configuration parameters
    """

    parser = argparse.ArgumentParser(description='Network models for characters recognition')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    # Input configuration
    parser.add_argument('--channels',
                        type=int,
                        default=1,
                        help='Input image channels')
    parser.add_argument('--height',
                        type=int,
                        default=32,
                        help='Input image height')
    parser.add_argument('--width',
                        type=int,
                        default=32,
                        help='Input image width')
    parser.add_argument('--thumb_height',
                        type=int,
                        default=32,
                        help='Input image maximum height before processing')
    parser.add_argument('--thumb_width',
                        type=int,
                        default=32,
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
    parser.add_argument('--geoletters',
                        dest='geoletters',
                        action='store_true',
                        default=False,
                        help='Flag for training on Georgian letters data.')
    parser.add_argument('--num_classes',
                        type=int,
                        default=33,
                        help='Number of output classes')
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
    flags.border = 0 if flags.make_borders else flags.border
    flags.labels = str(data_path() / 'models' / MODEL_LABELS)

    return flags
