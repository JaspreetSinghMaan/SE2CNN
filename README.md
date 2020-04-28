

Jaspreet Singh, Punjabi University, Patiala (Punjab, India)

28 April 2020

This DEMO was tested on a google colab.

The network is modified to be complete roto and tranlsation invariant by sticking with (5x5 and 1x1) SE(2) group convolutions all the way to the output layer, which would then provide a length 10 feature vector for each orientation. And then simply did a maximum projection over the orientations (tf.reduce_max) to get the maximal response for each bin, followed by a softmax.

This program is orginally written by Erik J. Bekkers and Maxime W. Lafarge, Eindhoven University of Technology, the Netherlands

8 June 2018

This DEMO was tested on a laptop with:

-Windows as OS

-Jupyter Notebook (version 5.5.0)

-Python (version 3.5.5)

-TensorFlow-GPU (versions 1.1 and higher)

-An NVIDIA Quadro M1000M GPU

-The following additional libraries installed for this demo to run: sklearn, scipy, and matplotlib

Basic usage of the se2cnn library

This jupyter demo will contain the basic usage examples of the se2cnn library with applications to digit recognition in the MNIST dataset. The se2cnn library contains 3 main layers (check the useage via help(se2cnn.layers.z2_se2n)):

    z2_se2n: a lifting layer from 2D tensor to SE(2) tensor
    se2n_se2n: a group convolution layer from SE(2) tensor to SE(2) tensor
    spatial_max_pool: performs 2x2 spatial max-pooling of the spatial axes

The following functions are used internally, but may be of interest as well:

    rotate_lifting_kernels: rotates the raw 2D lifting kernels (output is a set of rotated kernels)
    rotate_gconv_kernels: rotates (shift-twists) the se2 kernels (planar rotation + orientation shift

