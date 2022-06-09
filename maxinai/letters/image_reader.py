"""
Created on Jun 17, 2016
Image processing before recognition
@author: Levan Tsinadze
"""

import io

import PIL
from PIL import Image, ImageOps

_BORDER_COLOR = 'black'


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


def process_image_file(flags, img: PIL.Image):
    """Prepares image file
        Args:
            flags - configuration parameters
            img - binary image
        Returns:
            img_binary - converted image
    """

    dim = flags.thumb_width, flags.thumb_height
    img = img.convert("L")  # convert into grey-scale
    img = img.point(lambda i: i < 150 and 255)  # better black and white
    img = ImageOps.expand(
        img, border=flags.border, fill=_BORDER_COLOR)  # add padding
    rimg = _resize_algorithm(img, dim)
    img.thumbnail(dim, rimg)

    return img


def request_file(flags, image_data):
    """
    Read image from request
    Args:
        flags: configuration parameters
        image_data: image request

    Returns:
        proc_img: processed image
    """

    with Image.open(io.BytesIO(image_data)) as img:
        proc_img = process_image_file(flags, img)

    return proc_img
