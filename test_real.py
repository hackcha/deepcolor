import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.engine import Layer
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from skimage.transform import resize
import keras.activations
# from skimage.io import imsave
from matplotlib import pyplot
import numpy as np
import os
import random
import tensorflow as tf

r = np.loads( 'tmp.npz' )
curs, imgs = r['cur'], r['img']
