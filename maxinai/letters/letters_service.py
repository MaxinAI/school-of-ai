"""
Created on Nov 15, 2017

Service for model interface

@author: Levan Tsinadze
"""

import logging

from fastai.vision import *
from flask import (Flask, request, render_template, json)
from torch import (nn, Tensor, no_grad)
from torchvision import transforms

from maxinai.letters.image_reader import request_file
from maxinai.letters.service_config import configure
from path_utils import data_path

logger = logging.getLogger(__name__)

tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

_PREDICTION_KEY = 'prediction'

# Initializes web container
app = Flask(__name__)


class FlattenLayer(nn.Module):
    """Flatten layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, 1)


class ModelWrapper(object):
    """Model wrapper for inference"""

    def __init__(self, model: nn.Module, trfms: transforms):
        self.model = model.eval()
        self.trfms = trfms

    @no_grad()
    def forward(self, *imgs: PIL.Image) -> np.ndarray:
        itns = torch.stack([self.trfms(x) for x in imgs])
        otns = self.model(itns)
        results = otns.cpu().data.numpy()

        return results

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def conv2(ni: int, nf: int):
    return conv_layer(ni, nf, stride=2)


def conv_and_res(ni, nf):
    return nn.Sequential(conv2(ni, nf), res_block(nf))


def init_model():
    """
    Initialize model

    Returns:
        net: model with trained weights
    """
    net = nn.Sequential(
        conv_and_res(1, 32),
        conv_and_res(32, 64),
        conv_and_res(64, 128),
        conv_and_res(128, 256),
        conv_and_res(256, 512),
        conv2(512, 33),
        nn.AdaptiveAvgPool2d((1, 1)),
        FlattenLayer())
    #state_dict = torch.load(str(data_path() / 'models' / 'mnist_resnet.pth'), map_location='cpu')
    state_dict = torch.load(str(data_path() / 'models' / 'res_stage_1.pth'), map_location='cpu')
    weights_dict = state_dict['model']
    net.load_state_dict(weights_dict, strict=True)
    net.eval()

    return net


def init_wrapper():
    """
    Initialize model wrapper

    Returns:
        wrapper: model wrapper
    """
    net = init_model()
    wrapper = ModelWrapper(net, tfms)

    return wrapper


def recognize_image(image_data):
    """
    Recognizes from binary image
    Args:
        image_data: binary image

    Returns:
        response_json: prediction response
    """
    img = request_file(flags, image_data)
    predictions = model(img)
    predictions = np.argmax(predictions)

    response_dict = {'geoletters': 'true', _PREDICTION_KEY: class_names[predictions]}
    response_json = json.dumps(response_dict)

    return response_json


@app.route('/', methods=['GET', 'POST'])
def cnn_recognize():
    """Web method for recognition
      Returns:
        resp - recognition response
    """
    if request.method == 'POST':
        resp = recognize_image(request.data)
    elif request.method == 'GET':
        resp = render_template('index.html')

    return resp


@app.route('/upload', methods=['GET', 'POST'])
def cnn_upload():
    """Recognizes uploaded images
      Returns:
        resp - recognition response
    """
    return recognize_image(request.data)


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
    class_names = {int(idx): class_name for idx, class_name in labels_json.items()}
    logger.debug(class_names)

    return labels_json, class_names


if __name__ == "__main__":
    flags = configure()
    logging.basicConfig(level=logging.DEBUG if flags.verbose else logging.INFO)
    model = init_wrapper()
    _, class_names = load_labels(flags)
    flags.num_classes = len(class_names) if len(class_names) > 0 else flags.num_classes

    app.run(host=flags.host, port=flags.port, threaded=True)
