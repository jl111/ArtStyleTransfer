# -*- coding: utf-8 -*-

import os
import sys
import scipy.io
import random
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import PIL
import constants
import numpy as np
import tensorflow as tf

def get_output_name(content, style):
    return constants.OUTPUT_DIR + edit_output_name(content) + edit_output_name(style) + str(random.randint(1,101)) + ".jpg"

def prepareImage(imagePath):
    image = scipy.misc.imread(imagePath)
    return normalize_image(image)

def edit_output_name(filename):
    return filename.replace('images/','').replace('.jpg', "").replace('.png', "")

def gram(A):
    return tf.matmul(A, tf.transpose(A))

def get_noise_image(content_image, noise_ratio = constants.NOISE_RATIO, WIDTH = 400, HEIGHT = 300):
    
    noise_image = np.random.uniform(-20, 20, (1, HEIGHT, WIDTH, constants.COLOR_CHANNELS)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image

def normalize_image(image):

    image = np.reshape(image, ((1,) + image.shape))  - constants.MEANS    
    return image

def reshape_style_to_content(content, style):
    raveled = np.ravel(style)
    image = np.resize(raveled, (1, ) + content.shape)

    image = image - constants.MEANS

    return image

def resize_image_to_match(content, style):

    width, height = Image.open(content).size
    new_image = Image.open(style)
    new_image = new_image.resize((width, height), Image.ANTIALIAS)
    return new_image


def save_image(path, image):
    
    image = image + constants.MEANS
    
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)