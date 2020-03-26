"""
Created on Nov 13, 2016

Transfer learning files manager

@author: Levan Tsinadze
"""

from utils.files.file_utils import cnn_file_utils


class training_file(cnn_file_utils):
    """Files and directories for (trained),
       training, validation and test parameters"""

    def __init__(self):
        super(training_file, self).__init__('letters')


files = training_file()
