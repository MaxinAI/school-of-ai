"""
Created on Jun 17, 2016
Image processing before recognition
@author: Levan Tsinadze
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from utils.images.image_converter import convert_image_file

from maxinai.letters.cnn_files import files as _files

_BORDER_COLOR = 'black'

# Path to requested image log
_http_image_path = _files.data_file('http_img.png')


def _log_image(flags, img):
    """Saves image if debug mode is on
        Args:
            flags - configuration parameters
            img - image to save
    """

    if flags.verbose:
        img.save(_http_image_path)


def make_borders(img_tensor, flags):
    """Makes borders to input image tensor if configured
        Args:
            img_tensor - image tensor
            flags - configuration parameters
        Returns:
            img - padded image tensor
    """

    if flags.make_borders:
        img = cv2.copyMakeBorder(img_tensor,
                                 top=flags.border_px, bottom=flags.border_px,
                                 left=flags.border_px, right=flags.border_px,
                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img = img_tensor

    return img


def _resize_algorithm(img, dim):
    """Resize algorithm by input image size
        Args:
            img - input image
            dim - destination image size
        Returns:
            rimg - resize algorithm
    """

    if img is None:
        rimg = Image.BILINEAR
    elif img.size[0] > dim[0] or img.size[1] > dim[1]:
        rimg = Image.ANTIALIAS
    else:
        rimg = Image.BILINEAR

    return rimg


def _process_array(flags, img_array):
    """Processes image tensor
        Args:
            flags - configuration parameters
            img_array - image tensor
        Returns:
            img_binary - converted image
    """

    cnv_array = make_borders(img_array, flags)
    img_binary = Image.fromarray(cnv_array, mode='L')
    _log_image(flags, img_binary)

    return img_binary


def request_file(flags, image_data=None):
    """Reads image file to tensor
        Args:
            flags - configuration parameters
            image_data - binary image
        Returns:
            img - converted image tensor
    """

    dim = (flags.thumb_width, flags.thumb_height)
    with Image.open(io.BytesIO(image_data)) as img:
        img = img.convert("L")  # convert into grey-scale
        img = img.point(lambda i: i < 150 and 255)  # better black and white
        img = ImageOps.expand(img, border=flags.border, fill=_BORDER_COLOR)  # add padding
        rimg = _resize_algorithm(img, dim)
        img.thumbnail(dim, rimg)  # resize back to the same size
        img = np.array(img)
        img = _process_array(flags, img)

        return img


def process_image_file(flags, img):
    """Prepares image file
        Args:
            flags - configuration parameters
            img - binary image
        Returns:
            img_binary - converted image
    """

    dim = (flags.thumb_width, flags.thumb_height)
    img = img.convert("L")  # convert into grey-scale
    img_array = np.array(img)
    img_array = convert_image_file(img_array, dim, padding=flags.border_px)
    img_binary = _process_array(flags, img_array)

    return img_binary


def request_image_file(flags, image_data):
    """Converts binary image file for recognition
        Args:
            flags - configuration flags
            image_data - binary image
        Returns:
            img_array - converted image tensor
    """

    with Image.open(io.BytesIO(image_data)) as img:
        img_binary = process_image_file(flags, img)

    return img_binary


def process_image_path(flags, image_path):
    """Processes image from path
        Args:
            flags - configuration aprameters
            image_path - image path
        Returns:
            img_array - converted image tensor
    """

    with Image.open(image_path, 'r') as img:
        img_binary = process_image_file(flags, img)

    return img_binary
