"""
Created on Nov 15, 2017

Service for model interface

@author: Levan Tsinadze
"""

from flask import (Flask, request, render_template, json)

from maxinai.letters import (training_flags as config,
                             cnn_input_reader as reader,
                             letters_interface as interface)

_PREDICTION_KEY = 'prediction'

# Initializes web container
app = Flask(__name__)


def recognize_image(image_data, image_reader):
    """Recognizes from binary image
      Args:
        image_data - binary image
        image_reader - image reader function
      Returns:
        response_json - prediction response
    """

    img = image_reader(flags, image_data)
    predictions = interface.run_model(model, img)

    response_dict = {'geoletters': 'true', _PREDICTION_KEY: class_names[predictions]} \
        if flags.geoletters \
        else {_PREDICTION_KEY: predictions}
    response_json = json.dumps(response_dict)

    return response_json


def recognize_drawn():
    """Recognizes image index from request
      Returns:
        recognition response
    """
    return recognize_image(request.data, reader.request_file)


def recognize_file():
    """Recognizes image from file
      Returns:
        recognition response
    """

    upload_file = request.files['image-rec']
    if upload_file.filename:
        image_data = upload_file.read()
        response_json = recognize_image(image_data, reader.request_image_file)
    else:
        response_dict = {'error': 'File not found'}
        response_json = json.dumps(response_dict)

    return response_json


def cnn_recognize_method(page_name, recognize_method):
    """Recognize uploaded file
      Args:
        page_name - page name
      Returns:
        resp - recognition response
    """

    if request.method == 'POST':
        resp = recognize_method()
    elif request.method == 'GET':
        resp = render_template(page_name)

    return resp


@app.route('/', methods=['GET', 'POST'])
def cnn_recognize():
    """Web method for recognition
      Returns:
        resp - recognition response
    """
    return cnn_recognize_method("index.html", recognize_drawn)


@app.route('/upload', methods=['GET', 'POST'])
def cnn_upload():
    """Recognizes uploaded images
      Returns:
        resp - recognition response
    """
    return cnn_recognize_method("upload.html", recognize_file)


if __name__ == "__main__":
    """Starts letters model service"""

    global flags, model, class_names
    flags = config.configure()
    (model, _, class_names) = interface.init_model_and_labels(flags)

    app.run(host=flags.host, port=flags.port, threaded=True)
