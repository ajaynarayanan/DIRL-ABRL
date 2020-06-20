from __future__ import absolute_import

import os
import os.path as osp
import errno
import json
from collections import OrderedDict
import warnings
#import matplotlib.pyplot as plt
from PIL import Image


def read_rgb_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            got_img = True
        except IOError:
            print(
                "IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                    img_path
                )
            )
            pass
    return img


def show_image(image):
    dpi = 80
    figsize = (image.shape[1] / float(dpi), image.shape[0] / float(dpi))
    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    fig.show()


def get_file_name(filepath):
    return osp.split(filepath)[1]


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(path))
    return isfile


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))

