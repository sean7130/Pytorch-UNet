""" Utils on generators / lists of ids to transform from strings to cropped images and masks """

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (os.path.splitext(f)[0] for f in os.listdir(dir) if not f.startswith('.'))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        # print(suffix)
        if suffix == '_mask.gif' and (dir == "data/masks/" or dir == "data_cat/masks/"):
          # Case of using cat data masks
          # filename = "masks_"+id+".jpg"
          img_name_adjusted = dir + "mask_" + id + ".jpg"
          # print(img_name_adjusted)
          im = resize_and_crop(Image.open(img_name_adjusted), scale=scale)
          im = im/255
          yield im
        elif suffix == "_mask.gif" and dir == "cat_data_aug/masks/":
          img_name_adjusted = dir + "mask_" + id + ".jpg"
          im = resize_and_crop(Image.open(img_name_adjusted), scale=scale)
          im = im/255
          yield im 
        else:
          # any other case, for example using car data
          # filename = id+suffix, suffix is "_masks.jpg"
          im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
          yield im


def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)
    masks_switched = map(hwc_to_chw, masks)

    return zip(imgs_normalized, masks_switched)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
