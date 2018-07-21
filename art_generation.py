# -*- coding: utf-8 -*-
import os
import sys
import scipy.io
import scipy.misc

from PIL import Image
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import constants
from helpers import *
from VGG import *

STYLE_LAYERS = constants.STYLE_LAYERS
INPUT_DIR = constants.INPUT_DIR

def content_cost(a_C, a_G):
    
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))
    
    return tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)

def layer_style_cost(a_S, a_G):
    
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    GS = gram(tf.transpose(a_S))
    GG = gram(tf.transpose(a_G))

    return tf.reduce_sum((GS - GG) ** 2) / (4 * n_C ** 2 * (n_W * n_H) ** 2)


def style_cost(sess, model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out
        
        J_style += coeff * layer_style_cost(a_S, a_G)

    return J_style

def total_cost(content_cost, style_cost, alpha = 10, beta = 40):

    return alpha * content_cost + beta * style_cost

def trainModel(model, sess, input_image, train_step, num_iterations = 1):
    
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    # Logs process of the number of training iterations
    for i in tqdm(range(num_iterations)):

        _ = sess.run(train_step)

        generated_image = sess.run(model['input'])
    
    return generated_image


def start_model(sess, content, style):

    content_width, content_height = Image.open(content).size
    content_image = prepareImage(content)
    style_image = scipy.misc.imread(style)
    style_image = reshape_style_to_content(scipy.misc.imread(content), style_image)

    generated_image = get_noise_image(content_image, WIDTH = content_width, HEIGHT =content_height)

    model = build_CNN(content_width, content_height)  
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']

    a_C = sess.run(out)
    a_G = out

    sess.run(model['input'].assign(style_image))

    J = total_cost(content_cost(a_C, a_G), style_cost(sess, model, STYLE_LAYERS),  alpha = 10, beta = 40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    generated_image = trainModel(model, sess, generated_image, train_step, num_iterations = constants.ITER)

    save_image(get_output_name(content, style), generated_image)   

def blend_images(content, style):

    tf.reset_default_graph()

    sess = tf.InteractiveSession()
    content = INPUT_DIR + content
    style = INPUT_DIR + style

    start_model(sess, content, style)

def main():
    content = sys.argv[1]
    style = sys.argv[2]
    blend_images(content, style)

if __name__ == "__main__":
    main()


    


