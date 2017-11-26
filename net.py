import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.engine import Layer
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers import add
from keras.activations import relu
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
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU

def lrelu(x, leak = 0.2, name = "lrelu"):
    return K.maximum(x, leak * x)


def conv_stack(data, filters, s, activation=lrelu, padding='same',name=None):#padding='same'?
    x = Activation(activation)(data)
    x = Conv2D(filters, (5, 5), strides=s, padding=padding,
               kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name)(x)
    bn_name = None
    if name is not None:
        bn_name = name+'_bn'
    x = new_BatchNorm(bn_name)(x)
    # output = BatchNormalization()(output)
    return x

def conv_stack_noname(data, filters, s, activation=lrelu, padding='same',name=None):#padding='same'?
    x = Activation(activation)(data)
    x = Conv2D(filters, (5, 5), strides=s, padding=padding,
               kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name)(x)
    x = new_BatchNorm()(x)
    # output = BatchNormalization()(output)
    return x

def resnet_block( encoder_input, filter_size, stride ):
    x = conv_stack(encoder_input, filter_size, stride)
    x = conv_stack(x, filter_size, 1)
    shortcut = Conv2D(filter_size, (1, 1), strides=stride)(encoder_input)
    shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    return x

def abnet( img_size ):
    # Encoder
    encoder_input = Input(shape=(img_size, img_size, 1))
    encoder_output = conv_stack(encoder_input, 32, 2)
    encoder_output = conv_stack(encoder_output, 32, 1)

    encoder_output = conv_stack(encoder_output, 64, 2)
    encoder_output = conv_stack(encoder_output, 64, 1)
    encoder_output = conv_stack(encoder_output, 128, 2)
    encoder_output = conv_stack(encoder_output, 128, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)
    # Decoder
    decoder_output = conv_stack(encoder_output, 128, 1)
    decoder_output = conv_stack(decoder_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)  #
    # model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model = Model(inputs=encoder_input, outputs=decoder_output)
    model.summary()
    return model

bni = 0
def new_BatchNorm(name=None):
    return BatchNormalization(momentum=0.9,epsilon=1e-5, scale=True, name=name)

def rgbnet( img_size ):
    # Encoder
    name_prefix = "gen_"
    encoder_input = Input(shape=(img_size, img_size, 1))
    encoder_output = conv_stack(encoder_input, 32, 2, name=name_prefix+'c2')
    encoder_output = conv_stack(encoder_output, 32, 1)

    encoder_output = conv_stack(encoder_output, 64, 2)
    encoder_output = conv_stack(encoder_output, 64, 1)
    encoder_output = conv_stack(encoder_output, 128, 2)
    encoder_output = conv_stack(encoder_output, 128, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)
    # Decoder
    decoder_output = conv_stack(encoder_output, 128, 1)
    decoder_output = conv_stack(decoder_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_output)  #
    # model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model = Model(inputs=encoder_input, outputs=decoder_output)
    model.summary()
    return model

from keras.layers import Deconvolution2D, Conv2DTranspose
def rgb_unet( img_size, cdim = 4 ):
    # Encoder
    name_prefix = "gen_"
    # if as_tf:
    #     x_in = tf.placeholder(tf.float32, shape=[None, img_size, img_size, cdim])
    #     encoder_input = Input(tensor=x_in)
    # else:
    encoder_input = Input(shape=(img_size, img_size, cdim))
    df_dim = 64
    #e1 = conv_stack(encoder_input, df_dim, 2)#128
    e1 = Conv2D(df_dim, (5, 5), strides=2, padding='same',
               kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'c1')(encoder_input)
    e2 = conv_stack_noname(e1, df_dim*2, 2, name=name_prefix+'c2')#64
    e3 = conv_stack_noname(e2, df_dim*4, 2, name=name_prefix+'c3')#32
    e4 = conv_stack_noname(e3, df_dim*8, 2, name=name_prefix+'c4')#16
    # Decoder
    e5 = conv_stack_noname(e4, df_dim * 8, 2, name=name_prefix+'c5')#8

    d5 = Activation(lrelu)(e5)
    d4 = Conv2DTranspose(df_dim*8, (5,5), strides=2, padding='same',
                         kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'dc1')(d5)#16
    d4 = new_BatchNorm()(d4)
    d4 = concatenate([d4,e4], axis=-1)#64
    d4 = Activation(lrelu)(d4)
    d3 = Conv2DTranspose(df_dim * 2, (5, 5), strides=2, padding='same',
                         kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'dc2')(d4)  #32
    d3 = new_BatchNorm()(d3)
    d3 = concatenate([d3, e3], axis=-1)
    d3 = Activation(lrelu)(d3)
    d2 = Conv2DTranspose(df_dim , (5, 5), strides=2, padding='same',
                         kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'dc3')(d3)#64
    d2 = new_BatchNorm()(d2)
    d2 = concatenate([d2, e2],axis=-1)#256
    d2 = Activation(lrelu)(d2)
    d1 = Conv2DTranspose(df_dim, (5, 5), strides=2, padding='same',
                         kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'dc4')(d2)#128
    d1 = new_BatchNorm()(d1)
    d1 = concatenate([d1, e1], axis=-1)
    d1 = Activation(lrelu)(d1)

    decoder_output = Conv2DTranspose(3, (5, 5), strides=2, padding='same',
                            kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_prefix+'dc5' )(d1)  #256
    decoder_output = Activation('sigmoid')(decoder_output)
    # model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model = Model(inputs=encoder_input, outputs=decoder_output)
    return model

def discriminator( img_size, cdim = 3):
    name_frefix = "dis_"
    encoder_input = Input(shape=(img_size, img_size, cdim))
    df_dim = 64
    # e1 = conv_stack(encoder_input, df_dim, 2)
    #第一个不用BN
    e1 = Conv2D(df_dim, (5, 5), strides=2,
               kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_frefix+'c1')(encoder_input)
    e2 = conv_stack(e1, df_dim*2, 2,padding='valid', name=name_frefix+'c2')#64
    e3 = conv_stack(e2, df_dim*4, 2,padding='valid', name=name_frefix+'c3')#32
    e4 = conv_stack(e3, df_dim*8, 1,padding='valid', name=name_frefix+'c4')#32
    e4 = Activation(lrelu)(e4)
    d5 = Flatten()(e4 ) #32
    #instead of using activation='sigmoid', we can use cross_entropy_with_logits
    pred = Dense(1,kernel_initializer=keras.initializers.truncated_normal(stddev=0.02), name=name_frefix+'ds1' )(d5)
    model = Model(inputs=encoder_input, outputs=pred)
    return model

def CGAN( gm, dm, img_size, cdim=3 ):
    input = Input(shape=(img_size, img_size, cdim))
    generated_image = gm(input)
    feats = concatenate([generated_image, input], axis=-1)
    dcgan_output = dm(feats)
    dc_gan = Model(inputs=[input], outputs=[generated_image, dcgan_output], name="DCGAN")
    return dc_gan



def rgbnet_resnet(img_size):
    # Encoder
    encoder_input = Input(shape=(img_size, img_size, 1))
    encoder_output = resnet_block( encoder_input, 32, 2)
    encoder_output = resnet_block(encoder_output, 64, 2)
    encoder_output = resnet_block(encoder_output, 128, 2)
    # encoder_output = resnet_block(encoder_output, 256, 1)
    # Decoder
    decoder_output = resnet_block(encoder_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = resnet_block(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = resnet_block(decoder_output, 32, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_output)  #
    # model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model = Model(inputs=encoder_input, outputs=decoder_output)
    model.summary()
    return model




def abnet_resnet(img_size):
    # Encoder
    encoder_input = Input(shape=(img_size, img_size, 1))
    encoder_output = resnet_block( encoder_input, 32, 2)
    encoder_output = resnet_block(encoder_output, 64, 2)
    encoder_output = resnet_block(encoder_output, 128, 2)
    # encoder_output = resnet_block(encoder_output, 256, 1)
    # Decoder
    decoder_output = resnet_block(encoder_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = resnet_block(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = resnet_block(decoder_output, 32, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)  #
    # model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model = Model(inputs=encoder_input, outputs=decoder_output)
    model.summary()
    return model

def abnet_trans( img_size ):
    embed_input = Input(shape=(1000,))
    # Encoder
    encoder_input = Input(shape=(img_size, img_size, 1))
    # encoder_output = conv_stack(encoder_input, 32, 1)
    # encoder_output = conv_stack(encoder_output, 32, 1)
    # encoder_output = conv_stack(encoder_output, 64, 2)
    # encoder_output = conv_stack(encoder_output, 64, 1)
    # TODO 对于256 size的图像
    encoder_output = conv_stack(encoder_input, 64, 2)
    encoder_output = conv_stack(encoder_output, 128, 1)
    encoder_output = conv_stack(encoder_output, 128, 2)
    encoder_output = conv_stack(encoder_output, 256, 1)
    encoder_output = conv_stack(encoder_output, 256, 2)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 512, 1)
    encoder_output = conv_stack(encoder_output, 256, 1)

    # Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([fusion_output, encoder_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output)

    # Decoder
    decoder_output = conv_stack(fusion_output, 128, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = conv_stack(decoder_output, 64, 1)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = conv_stack(decoder_output, 32, 1)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)  #
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    model.summary()
    return model

from dc_main import *
def generator(img_in, batch_size=4):
    output_size = 256
    gf_dim  = 64
    s = output_size
    s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
    # image is (256 x 256 x input_c_dim)
    e1 = conv2d(img_in, gf_dim, name='g_e1_conv') # e1 is (128 x 128 x self.gf_dim)
    e2 = bn(conv2d(lrelu(e1), gf_dim*2, name='g_e2_conv')) # e2 is (64 x 64 x self.gf_dim*2)
    e3 = bn(conv2d(lrelu(e2), gf_dim*4, name='g_e3_conv')) # e3 is (32 x 32 x self.gf_dim*4)
    e4 = bn(conv2d(lrelu(e3), gf_dim*8, name='g_e4_conv')) # e4 is (16 x 16 x self.gf_dim*8)
    e5 = bn(conv2d(lrelu(e4), gf_dim*8, name='g_e5_conv')) # e5 is (8 x 8 x self.gf_dim*8)


    d4, d4_w, d4_b = deconv2d(tf.nn.relu(e5), [batch_size, s16, s16, gf_dim*8], name='g_d4', with_w=True)
    d4 = bn(d4)
    d4 = tf.concat([d4, e4],3)
    # d4 is (16 x 16 x self.gf_dim*8*2)

    d5, d5_w, d5_b = deconv2d(tf.nn.relu(d4), [batch_size, s8, s8, gf_dim*4], name='g_d5', with_w=True)
    d5 = bn(d5)
    d5 = tf.concat([d5, e3],3)
    # d5 is (32 x 32 x self.gf_dim*4*2)

    d6, d6_w, d6_b = deconv2d(tf.nn.relu(d5), [batch_size, s4, s4, gf_dim*2], name='g_d6', with_w=True)
    d6 = bn(d6)
    d6 = tf.concat([d6, e2],3)
    # d6 is (64 x 64 x self.gf_dim*2*2)

    d7, d7_w, d7_b = deconv2d(tf.nn.relu(d6), [batch_size, s2, s2, gf_dim], name='g_d7', with_w=True)
    d7 = bn(d7)
    d7 = tf.concat([d7, e1],3)
    # d7 is (128 x 128 x self.gf_dim*1*2)

    d8, d8_w, d8_b = deconv2d(tf.nn.relu(d7), [batch_size, s, s, 3], name='g_d8', with_w=True)
    # d8 is (256 x 256 x output_c_dim)
    return tf.nn.sigmoid(d8)#tf.nn.tanh(d8)

def discriminator_tf( image, d_bn1, d_bn2, d_bn3, reuse=False,batch_size=4):
    # image is 256 x 256 x (input_c_dim + output_c_dim)
    if reuse:
        tf.get_variable_scope().reuse_variables()
    else:
        assert tf.get_variable_scope().reuse == False
    df_dim = 64
    h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv')) # h0 is (128 x 128 x self.df_dim)
    h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv'))) # h1 is (64 x 64 x self.df_dim*2)
    h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv'))) # h2 is (32 x 32 x self.df_dim*4)
    h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, d_h=1, d_w=1, name='d_h3_conv'))) # h3 is (16 x 16 x self.df_dim*8)
    h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h3_lin')
    return tf.nn.sigmoid(h4), h4