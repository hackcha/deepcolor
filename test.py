import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
# from skimage.io import imsave
from matplotlib import pyplot
import numpy as np
import os
import random
import tensorflow as tf

files = ['0Bwa2J.jpg','0bzkK4.jpg']
X = []
for filename in files:
    X.append(img_to_array(load_img('data/Test/' + filename)))
X = np.array(X, dtype=float)
pyplot.imsave("data/result/img_"+str(0)+".jpg", X[0].astype( 'uint8' ))
# Xtrain = 1.0*X
Xtrain = X
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        # shear_range=0.4,
        # zoom_range=0.4,
        # rotation_range=40,
        horizontal_flip=True)
def my_rgb_to_gray( rgb ):
    gray = 0.2125 * rgb[..., 0]
    gray[:] += 0.7154 * rgb[..., 1]
    gray[:] += 0.0721 * rgb[..., 2]
    # gray = 0.299 * rgb[..., 0]
    # gray[:] += 0.587 * rgb[..., 1]
    # gray[:] += 0.114 * rgb[..., 2]
    return gray

index = 1
for imgs in datagen.flow( Xtrain, batch_size=1):
    index += 1
    #gray2rgb, rgb2gray不需要除以255.0
    print(imgs.dtype)
    g_im = my_rgb_to_gray( imgs )
    # print(g_im[-1])
    # print('***********')
    #concatenace( 3 * (image,) )
    grayscaled_rgb = gray2rgb(g_im)
    # print( grayscaled_rgb[-1] )
    pyplot.imsave("data/result/img_"+str(index)+".jpg", grayscaled_rgb[-1].astype( 'uint8' ))
    resize_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        img_ = resize(img, (224, 224, 3), mode='constant', preserve_range=True)
        resize_imgs.append(img_)
    resize_imgs = np.array(resize_imgs)
    print(resize_imgs[-1])
    # resize_imgs = preprocess_input(resize_imgs)
    pyplot.imsave("data/result/imgrs_" + str(index) + ".jpg", resize_imgs[-1].astype( 'uint8' ) )
    #在使用rgb2lab之前，必须除以255.0
    lab = rgb2lab(imgs/255.0)
    print(lab)
    recover = (lab2rgb(lab[-1]))*255.0
    print(recover)
    pyplot.imsave("data/result/imgre_"+str(index)+".jpg", recover.astype('uint8'))
    if index>5:
        break