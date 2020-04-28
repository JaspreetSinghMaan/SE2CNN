

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


