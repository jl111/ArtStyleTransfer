
# -*- coding: utf-8 -*-
import numpy as np

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

COLOR_CHANNELS = 3
NOISE_RATIO = 0.6
MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
VGG_MODEL = './imagenet-vgg-verydeep-19.mat'
OUTPUT_DIR = 'output/'
INPUT_DIR = 'images/'
ALPHA = 20
BETA = 50

ITER = 200