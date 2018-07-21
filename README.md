# ArtStyleTransfer
## Overview 
A simple deep convolutional neural network to transfer styles from one image onto another using TensorFlow in Python

My network follows the architecture of the VGG network and uses the weights from imagenet-vgg-verydeep-19.mat.

The style layers, and loss functions follow the implenetation of the network from the paper "Neural Algorithm of Artistic Style" (Gatys,Ecker,Bethge). 


## Example  


## How to run
You must have an "images" and an "output" folder.
Images will be read from "images" and then saved in "output"

$ python art_generation.py "CONTENT_IMAGE.jpg" "STYLE_IMAGE.jpg" 





