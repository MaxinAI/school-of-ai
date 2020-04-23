"""
Created on Nov 15, 2017

Network model interface for letres

@author: Levan Tsinadze
"""

import numpy as np
import torch
from torchvision import transforms
from utils.files import file_utils

from maxinai.letters import cnn_input_reader as reader
from maxinai.letters.dataset_utils import read_labels
from maxinai.letters.network_model import choose_model
from maxinai.letters.train_network import validate_test
from utils.logging import logger

# Functions for input tensor pre-processing
transform_func = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])


def to_variable(input_tensor):
    """Converts input tensor to variable
      Args:
        input_tensor - input image tensor
      Returns:
        x - variable for network model
    """
    x = transform_func(input_tensor)
    x = x.unsqueeze(0) if len(x.size()) <= 3 else x

    return x


def _extract_result(result_data):
    """Extract result from array
      Args:
        result_data - result array
      Returns:
        result_val  - extracted value
    """

    result_array = result_data[:1][0]
    result_val = np.argmax(result_array)

    return result_val


def process_output(model_output):
    """Process network model output
      Args:
        model_output - network model output
      Returns:
        result_val - output as standard data type
    """

    result_array = model_output.data.cpu().numpy()
    result_val = _extract_result(result_array)

    return result_val


def run_batch(model, input_batch):
    """Runs model on batch
      Args:
        model - network model
        input_batch - batch of input tensors
      Returns:
        result_array - result of model
    """

    x = to_variable(input_batch)
    output = model(x)
    result_array = output.data.cpu().numpy()

    return result_array


def run_model(model, input_image):
    """Runs model interface
      Args:
        model - network model
        input_image - input image tensor
      Returns:
        result_val - prediction result
    """

    result_array = run_batch(model, input_image)
    result_val = _extract_result(result_array)

    return result_val


def run_model_on_batch(model, input_image):
    """Runs model interface
      Args:
        model - network model
        input_image - input image tensor
      Returns:
        result_val - prediction result
    """

    x = to_variable(input_image)
    output = model(x)
    result_array = output.data.cpu().numpy()
    result_val = _extract_result(result_array)

    return result_val


def init_model(flags):
    """Initializes appropriated network model from configuration
      Args:
        flags - configuration parameters
      Returns:
        model - initialized appropriated network model
    """

    model = choose_model(flags)
    model.load_state_dict(torch.load(flags.weights, map_location=lambda storage, loc: storage))
    model.eval()
    logger.log(flags, model)

    return model


def load_labels(flags):
    """Reads labels JSON file
      Args:
        flags - configuration parameters
      Returns:
        tuple of -
          labels_json - labels JSON with indices
          class_names - class labels
    """

    labels_json = read_labels(flags)
    class_names = {int(idx): class_name for (idx, class_name) in labels_json.iteritems()}
    logger.log(flags, class_names)

    return labels_json, class_names


def init_model_and_labels(flags):
    """Reads model and labels
      Args:
        flags - configuration parameters
      Returns:
        tuple of -
          model - network model
          labels_json - labels JSON with indices
          class_names - class labels
    """

    (labels_json, class_names) = load_labels(flags)
    flags.num_classes = len(class_names) if len(class_names) > 0 else flags.num_classes
    model = init_model(flags)

    return model, labels_json, class_names


def run_on_directory(flags):
    """Runs test on directory of images
      Args:
        flags - configuration parameters
    """

    (model, _, _) = init_model_and_labels(flags)
    imgs = []
    for dir_name in file_utils.list_subdirs(flags.dir):
        for file_path in file_utils.list_files(dir_name, file_exts=flags.exts):
            img = reader.process_image_path(flags, file_utils.join(dir_name, file_path))
            imgs.append((img, int(dir_name)))
    validate_test(imgs, model, flags, testing=True)
