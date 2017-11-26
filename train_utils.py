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
import cv2
from random import randint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import utils

"""
修改了加载图片的函数
img_to_array 可能有问题
load_img 可能没有问题
"""


def set_keras_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    set_session(sess)

def create_inception_embedding( grayscaled_rgb, inception):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        #299 for inception, 224 for vgg
        i = resize(i, (299, 299, 3), mode='constant', preserve_range=True)
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

def load_inception():
    inception = InceptionV3(weights='imagenet', include_top=True)
    inception.graph = tf.get_default_graph()
    return inception

def get_Xtrain():
    X = []
    for filename in os.listdir('data/images/Train/'):
        X.append(img_to_array(load_img('data/images/Train/' + filename)))
    X = np.array(X, dtype=float)
    Xtrain = 1.0 / 255 * X
    print('load data done.')
    return Xtrain

def get_Xtrainlimit(img_size, data_dir='data/images/Train/', limit=512):
    X = []
    files = os.listdir(data_dir)
    for filename in files[:limit]:
        X.append(utils.get_image(data_dir + filename, target_size=(img_size, img_size)))
    Xtrain = np.array(X)
    return Xtrain

def image_ab_gen_trans(inception, datagen, Xtrain, batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(my_rgb_to_gray(batch))
        embed = create_inception_embedding(inception, grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128.0
        yield ([X_batch, embed], Y_batch)

def image_a_b_gen_batches(all_files, batch_size, img_size, trans=False, inception=None, data_dir ='data/images/Train/' ):
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    #Get images
    while True:
        np.random.shuffle( all_files )
        for bi in range( int(len(all_files)/batch_size) ):
            files = all_files[bi*batch_size:(bi+1)*batch_size]
            X = []
            for filename in files:
                X.append(img_to_array(load_img( data_dir+ filename,
                                               target_size=(img_size, img_size))))
            Xtrain = np.array(X, dtype=float)
            batch_index = 0
            for batch in datagen.flow(Xtrain, batch_size=batch_size):
                batch_index = batch_index + 1
                if batch_index > 1:
                    break
                # for batch in datagen.flow(Xtrain, batch_size=batch_size):
                lab_batch = rgb2lab(batch / 255.0)
                X_batch = lab_batch[:, :, :, 0]
                X_batch = X_batch.reshape(X_batch.shape + (1,))
                Y_batch = lab_batch[:, :, :, 1:] / 128.0
                if trans:
                    gray_imgs = my_rgb_to_gray(batch)
                    grayscaled_rgb = gray2rgb(gray_imgs)
                    embed = create_inception_embedding(grayscaled_rgb, inception)
                    yield ([X_batch, embed], Y_batch)
                else:
                    yield (X_batch, Y_batch)

def image_ab_gen(datagen, Xtrain, batch_size, trans= False, inception = None):
    # Get images
    # X = []
    # for filename in files:
    #     X.append(img_to_array(load_img('data/images/Train/' + filename, target_size=(img_size, img_size))))
    # Xtrain = np.array(X, dtype=float)
    ii = 0
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        ii += 1
        lab_batch = rgb2lab( batch/255.0 )
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128.0
        #测试发现是正常的
        # img_size = 256
        # cur = np.zeros((img_size, img_size, 3), dtype=np.float64)
        # Y = Y_batch *128
        # re = Y[0]
        # cur[:, :, 0] = X_batch[0][:, :, 0]
        # cur[:, :, 1:] = re
        # img = lab2rgb(cur) * 255.0
        # pyplot.imsave("data/result/imgflow_" + str(ii) + ".jpg", img.astype('uint8'))
        if trans:
            gray_imgs = my_rgb_to_gray(batch)
            grayscaled_rgb = gray2rgb(gray_imgs)
            embed = create_inception_embedding(grayscaled_rgb, inception)
            yield ([X_batch, embed], Y_batch)
        else:
            yield (X_batch, Y_batch)

def image_ab_valid(datagen, Xtrain, batch_size, trans= False, inception = None):
    ii = 0
    ii += 1
    lab_batch = rgb2lab( Xtrain/255.0 )
    X_batch = lab_batch[:,:,:,0]
    X_batch = X_batch.reshape(X_batch.shape+(1,))
    Y_batch = lab_batch[:,:,:,1:] / 128.0
    # Y_batch = Xtrain/255.0
    if trans:
        gray_imgs = my_rgb_to_gray(Xtrain)
        grayscaled_rgb = gray2rgb(gray_imgs)
        embed = create_inception_embedding(grayscaled_rgb, inception)
        return ([X_batch, embed], Y_batch)
    return (X_batch, Y_batch)



def image_rgb_gen(datagen, Xtrain, batch_size):
    ii = 0
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        ii += 1
        lab_batch = rgb2lab( batch/255.0 )
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        # Y_batch = lab_batch[:,:,:,1:] / 128.0
        Y_batch = batch/255.0
        yield (X_batch, Y_batch)

def image_rgb_valid(datagen, Xtrain, batch_size, do_blur=False):
    return  get_rgb_XY( Xtrain, do_blur)




def imageblur( cimg, sampling=False):
    """
    #这个操作会改变图像
    :param cimg:
    :param sampling:
    :return:
    """
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in range(30):
            randx = randint(0,205)
            randy = randint(0,205)
            cimg[randx:randx+50, randy:randy+50] = 255
    return cv2.blur(cimg,(100,100))

def get_rgb_XY( batch, do_blur, return_edge = False ):
    Y_batch = batch / 255.0
    if do_blur:
        add_edge = True
        if add_edge:
            batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                         blockSize=9,
                                                         C=2) for ba in batch]) / 255.0
            batch_edge = np.expand_dims(batch_edge, 3)
            base_colors = np.array([imageblur(ba) for ba in batch]) / 255.0
            X_batch = np.concatenate([ base_colors, batch_edge], axis=3)#X_batch
        else:
            lab_batch = rgb2lab(batch / 255.0)
            X_batch = lab_batch[:, :, :, 0] / 100.0
            X_batch = X_batch.reshape(X_batch.shape + (1,))
            base_colors = np.array([imageblur(ba) for ba in batch]) / 255.0
            X_batch = np.concatenate([X_batch, base_colors], axis=3)
    else:
        lab_batch = rgb2lab(batch / 255.0)
        X_batch = lab_batch[:, :, :, 0] / 100.0
        X_batch = X_batch.reshape(X_batch.shape + (1,))
    if return_edge:
        return X_batch, Y_batch, batch_edge, base_colors
    return (X_batch, Y_batch)

def image_rgb_gen_batches(all_files, batch_size, img_size, do_blur=False, train_data_dir='data/images/Train/'):
    #Get images
    # datagen = ImageDataGenerator(
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     rotation_range=20,
    #     horizontal_flip=True)
    while True:
        np.random.shuffle( all_files )
        for bi in range( int(len(all_files)/batch_size) ):
            files = all_files[bi*batch_size:(bi+1)*batch_size]
            X = []
            for filename in files:
                X.append(utils.get_image( train_data_dir+ filename,
                                               target_size=(img_size, img_size)))
            Xtrain = np.array(X)
            yield  get_rgb_XY(Xtrain, do_blur)



def my_rgb_to_gray(rgb):
    gray = 0.2125 * rgb[..., 0]
    gray[:] += 0.7154 * rgb[..., 1]
    gray[:] += 0.0721 * rgb[..., 2]
    # gray = 0.299 * rgb[..., 0]
    # gray[:] += 0.587 * rgb[..., 1]
    # gray[:] += 0.114 * rgb[..., 2]
    return gray

def load_test( img_size, data_dir='data/Test/' ):
    imgs = []
    for filename in os.listdir(data_dir)[-4:]:
        imgs.append(utils.get_image(data_dir + filename, target_size=(img_size, img_size)))
    imgs = np.array(imgs)
    gray_me = gray2rgb(my_rgb_to_gray(imgs))
    # TODO 需要进行转化
    # color_me = 1.0/255*color_me
    # color_me_embed = create_inception_embedding(gray_me)
    color_me = rgb2lab(1.0 / 255 * gray_me)[:, :, :, 0]/100.0
    color_me = color_me.reshape(color_me.shape + (1,))
    return imgs, color_me