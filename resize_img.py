import numpy
import cv2

def resize_img(input_img, output_size=(128,128), preserve_proportion=False):
    if preserve_proportion:
        dim = min(input_img.shape[0:2])
        input_img = corp_img(input_img, dim, dim)
    return cv2.resize(input_img, output_size)

def corp_img(input_img, new_height, new_width):
    ''' Force corp image though the center
    '''
    img_h, img_w = input_img.shape[0:2]
    # Determine the start and end, for x and y dir
    new_start_height = (img_h - new_height)//2
    new_start_width = (img_w - new_width)//2
    return input_img[new_start_height:new_start_height+new_height, 
            new_start_width:new_start_width+new_width,
            :]

