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
import keras.losses
import keras.activations
# from skimage.io import imsave
from matplotlib import pyplot
import numpy as np
import os
import random
import tensorflow as tf
import sys
from train_utils import *
from net import *
from keras.utils import generic_utils as keras_generic_utils
upsplash = False

if upsplash:
    model_dir = 'keras'
    test_data_dir = "data/images/Test/"
    train_data_dir = "data/images/Train/"
    result_dir = "data/result/"
else:
    model_dir = 'comic_k'
    test_data_dir = "imgs/"
    train_data_dir = "imgs/"
    result_dir = "results/"
ckpt_dir = "checkpoint/"
img_size = 256
batch_size = 4
def ab_main( trainit = True, cont = False):
    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    # Generate training data
    model = abnet_resnet(img_size)
    # Train model
    filepath = "color_model_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, period=1,
                                 save_best_only=True)
    callbacks_list = [checkpoint]
    opt = SGD(lr=0.0005, momentum=0.0, clipnorm=5.0, decay=0.999)
    # opt = Adam(lr=0.001, clipnorm=5.0, epsilon=0.1)
    model.compile(optimizer=opt, loss='mse')
    if cont:
        model.load_weights( filepath )
    all_files = os.listdir('data/images/Train/')
    steps_per_epoch = 512
    train_it = True
    if train_it:
        Xtest = get_Xtrainlimit(img_size, data_dir='data/Test/', limit=64)
        # Xtrain = get_Xtrainlimit(img_size)
        # model.fit_generator(image_ab_gen(datagen, Xtrain, batch_size), epochs=1,
        #                     steps_per_epoch=int(Xtrain.shape[0]/batch_size),
        #                     callbacks=callbacks_list, verbose=1)
        model.fit_generator(image_a_b_gen_batches(all_files, batch_size, img_size),
                            validation_data= image_ab_valid(datagen, Xtest, batch_size),
                            epochs=20, steps_per_epoch=int(len(all_files)/batch_size),
                            callbacks=callbacks_list, verbose=1)
        model.save_weights("color_model.h5" )
    # model.load_weights( filepath )
    # Make predictions on validation images
    if not train_it and (not cont):
        model.load_weights(filepath)
    imgs, color_me = load_test(img_size, data_dir='data/testdata/Validate/')
    # Test model
    # output = model.predict([color_me/100.0, color_me_embed])
    output = model.predict(color_me)
    output = output * 128.0
    # Output colorizations
    curs = []
    for i in range(len(output)):
        cur = np.zeros((img_size, img_size, 3), dtype=np.float64)
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        curs.append(cur)
        img = lab2rgb(cur) * 255.0
        # pylo
        pyplot.imsave("data/result/img_" + str(i) + ".jpg", img.astype('uint8'))
        # pyplot.imsave("data/result/imgo_" + str(i) + ".jpg", imgs[i].astype( 'uint8' )  )

def ab_trans_main(train_it = True, cont = False):
    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    # Generate training data
    model = abnet_trans(img_size)
    # Train model
    filepath = "ab_trans_model_best.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, period=1,
                                 save_best_only=True)
    callbacks_list = [checkpoint]
    # opt = SGD(lr=0.0005, momentum=0.0, clipnorm=50.0, decay=1e-5)
    opt = Adam(lr=0.001, beta_2=0.5, clipnorm=50.0, epsilon=0.01)#0.001可以缓慢的学习
    model.compile(optimizer=opt, loss='mse')
    filepath = "ab_trans_model.h5"
    if cont:
        model.load_weights( filepath )
    all_files = os.listdir(train_data_dir)
    steps_per_epoch = 512
    # train_it = False
    inception = load_inception()
    if train_it:
        Xtest = get_Xtrainlimit(img_size, data_dir=test_data_dir, limit=64)
        # Xtrain = get_Xtrainlimit(img_size)
        # model.fit_generator(image_ab_gen(datagen, Xtrain, batch_size), epochs=1,
        #                     steps_per_epoch=int(Xtrain.shape[0]/batch_size),
        #                     callbacks=callbacks_list, verbose=1)
        model.fit_generator(image_a_b_gen_batches(all_files, batch_size, img_size, trans=True, inception=inception, data_dir=train_data_dir),
                            validation_data= image_ab_valid(datagen, Xtest, batch_size, trans=True, inception=inception),
                            epochs=20000, steps_per_epoch=int(len(all_files)/batch_size),
                            callbacks=callbacks_list, verbose=1)
        model.save_weights(filepath )
    if not train_it and (not cont):
        model.load_weights( filepath )
    # Make predictions on validation images
    # model.load_weights(filepath)
    imgs, color_me = load_test(img_size, data_dir=test_data_dir)
    gray_imgs = my_rgb_to_gray(imgs)
    grayscaled_rgb = gray2rgb(gray_imgs)
    embed = create_inception_embedding(grayscaled_rgb, inception)
    # Test model
    # output = model.predict([color_me/100.0, color_me_embed])
    output = model.predict([color_me, embed])
    output = output * 128.0
    # Output colorizations
    curs = []
    for i in range(len(output)):
        cur = np.zeros((img_size, img_size, 3), dtype=np.float64)
        cur[:, :, 0] = color_me[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        curs.append(cur)
        img = lab2rgb(cur) * 255.0
        # pylo
        pyplot.imsave(result_dir+"img_" + str(i) + ".jpg", img.astype('uint8'))
        # pyplot.imsave("data/result/imgo_" + str(i) + ".jpg", imgs[i].astype( 'uint8' )  )


def rgb_gen( model, epoch, do_blur=False):
    imgs, color_me = load_test(img_size, data_dir=test_data_dir)
    color_me, _ = get_rgb_XY( imgs, do_blur)
    output = model.predict(color_me)
    # Output colorizations
    lab_batch = rgb2lab(output)
    size = int( np.ceil( np.sqrt( len(output) ) ) )
    img = utils.merge_color( output, (size, size)) * 255
    # pyplot.imsave("data/result/img_" + str(epoch) + ".jpg", img.astype('uint8'))
    cv2.imwrite(result_dir+"img_" + str(epoch) + ".jpg", img)

def rgb_gen_tf( sess, g_in, real_images, gen_img, epoch, do_blur=False):
    imgs, _ = load_test(img_size, data_dir=test_data_dir)
    color_me, imgs, batch_edge, base_colors = get_rgb_XY( imgs, do_blur, return_edge=True)
    output = sess.run(gen_img,
                          feed_dict={real_images: imgs, g_in: color_me,K.learning_phase():0})
    # print(output)
    # print(output.dtype)
    size = int( np.ceil( np.sqrt( len(output) ) ) )
    img = utils.merge_color( output, (size, size))*255.0
    tar = utils.merge_color(imgs, (size, size)) * 255.0
    cv2.imwrite(result_dir+"img_" + str(epoch) + ".jpg", img)
    cv2.imwrite(result_dir + "target_" + str(epoch) + ".jpg", tar)

def rgb_main( trainit = True, cont=False):
    # Image transformer
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
    # Generate training data
    do_blur = True
    cdim = 1
    if do_blur:
        cdim = 4
    model = rgb_unet(img_size, cdim = cdim)
    # Train model
    filepath = "rgb_unet4_best_{}.h5".format(do_blur)
    filepath_last = "rgb_unet4_{}.h5".format(do_blur)
    tb_cbk = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=batch_size, write_graph=False)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, period=1,
                                 save_best_only=True)
    callbacks_list = [checkpoint, tb_cbk]
    # opt = SGD(lr=0.01, momentum=0.0)
    #0.003还是损失函数下降很慢
    #开始是0.001
    opt = Adam(lr=0.0001, beta_2=0.5)#0.001, epsilon=0.01 clipnorm=50.0,, decay=0.001
    model.compile(optimizer=opt, loss='binary_crossentropy')
    #继续训练
    if cont:
        model.load_weights( filepath )
        print('load weight done.')
    all_files = os.listdir(train_data_dir)
    steps_per_epoch = 512
    if trainit:
        for epoch in range( 100 ):
            Xtest = get_Xtrainlimit(img_size, data_dir=test_data_dir,limit=64)
            # Xtrain = get_Xtrainlimit(img_size)
            # model.fit_generator(image_rgb_gen(datagen, Xtrain, batch_size), epochs=20,
            #                     steps_per_epoch=int(Xtrain.shape[0]/batch_size),
            #                     validation_data=image_rgb_valid(datagen, Xtest, batch_size),
            #                     callbacks=callbacks_list, verbose=1)
            model.fit_generator(image_rgb_gen_batches(all_files, batch_size, img_size, do_blur=do_blur, train_data_dir=train_data_dir),
                                validation_data=image_rgb_valid(datagen, Xtest, batch_size, do_blur=do_blur),
                                epochs=1, steps_per_epoch=int(len(all_files)/batch_size),
                                callbacks=callbacks_list, verbose=1)
            rgb_gen( model, epoch, do_blur=do_blur )
            model.save_weights(filepath_last)
    else:
        if not cont:
            model.load_weights(filepath)
        rgb_gen(model, 0, do_blur=do_blur)


def save_model_tf( ckpt_dir, sess, saver, step):
    model_name = "model"
    checkpoint_dir = os.path.join(ckpt_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir, model_name),
                    global_step=step)

def load_model_tf( ckpt_dir, sess, saver ):
    checkpoint_dir = os.path.join(ckpt_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Loaded.')
    else:
        print("Load failed")


def rgb_gan_main( trainit = True, cont=False, load_discrim=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    set_session(sess)
    # Generate training data
    do_blur = True
    cdim = 1
    if do_blur:
        cdim = 4
    # gm  = rgb_unet(img_size, cdim = cdim)
    # gm.summary()
    # dm= discriminator(img_size, cdim=cdim+3)
    # dm.summary()
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')
    g_in = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, cdim])
    real_images = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 3])
    # gen_img = gm(g_in)
    gen_img = generator(g_in)
    real_AB = tf.concat([g_in, real_images], 3)
    fake_AB = tf.concat([g_in, gen_img], 3)
    # disc_true_logits = dm(real_AB)
    # disc_fake_logits = dm(fake_AB)
    disc_true_logits = discriminator_tf(real_AB, d_bn1, d_bn2, d_bn3, reuse=False)
    disc_fake_logits = discriminator_tf(fake_AB, d_bn1, d_bn2,d_bn3, reuse=True)


    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
    d_loss = d_loss_real + d_loss_fake
    mae_loss = tf.reduce_mean(tf.abs(real_images - gen_img))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                  + 100.0 * mae_loss
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    d_var_names = [var.name for var in d_vars]
    g_var_names = [var.name for var in g_vars]
    print('\n'.join(d_var_names) )
    print('\n'.join(g_var_names) )
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)
    sess.run(tf.global_variables_initializer() )
    if load_discrim:
        saver = tf.train.Saver()
    else:
        saver = tf.train.Saver(g_vars)
    # Train model
    #继续训练
    if cont:
        load_model_tf(ckpt_dir,sess, saver)
    all_files = os.listdir(train_data_dir)
    steps_per_epoch = 512

    rgb_gen_tf(sess, g_in, real_images, gen_img, 10000, do_blur=do_blur)
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.name, var)
    if trainit:
        imgs, _ = load_test(img_size, data_dir=test_data_dir)
        color_me, imgs, batch_edge, base_colors = get_rgb_XY(imgs, do_blur, return_edge=True)
        size = int(np.ceil(np.sqrt(len(imgs))))
        ri = utils.merge_color(imgs, (size, size)) * 255.0
        edge = utils.merge_color( batch_edge, (size, size))*255.0
        color = utils.merge_color( base_colors, (size, size))*255.0
        cv2.imwrite(result_dir + "base.png", color)
        cv2.imwrite(result_dir + "base_line.jpg", edge)
        cv2.imwrite(result_dir + "real_img.jpg", ri)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('logs',
        #                                      sess.graph)
        all_batches = image_rgb_gen_batches(all_files, batch_size, img_size, do_blur=do_blur,
                                            train_data_dir=train_data_dir)
        for epoch in range( 20000 ):
            # Xtest = get_Xtrainlimit(img_size, data_dir=test_data_dir,limit=64)
            progbar = keras_generic_utils.Progbar( 100*batch_size )#len(all_files)
            batch_epoch = 100#int(len(all_files)/batch_size)
            batch_counter = 0
            for batch_counter in range( batch_epoch ):
                batch = next(all_batches)
                disc_loss, gen_mae, _ = sess.run([d_loss, mae_loss, d_optim],
                                          feed_dict={real_images: batch[1], g_in: batch[0],K.learning_phase():1 })

                gen_loss, _ = sess.run([g_loss, g_optim],
                                          feed_dict={real_images: batch[1], g_in: batch[0],K.learning_phase():1 })

                gen_total_loss = min(gen_loss+gen_mae, 10000)
                gen_mae = min(gen_mae,10000)
                gen_log_loss = min(gen_loss, 10000)
                progbar.add(batch_size, values=[("D_l", disc_loss),
                                                ("G_lt", gen_total_loss),
                                                ("G_mae", gen_mae),
                                                ("G_l", gen_log_loss)])
                if batch_counter == batch_epoch-1:
                    r_i = utils.merge_color(batch[1], (size, size)) * 255.0
                    output = sess.run(gen_img,
                                      feed_dict={real_images: batch[1], g_in: batch[0], K.learning_phase(): 0})
                    o_i = utils.merge_color(output, (size, size)) * 255.0
                    cv2.imwrite(result_dir + "real_img_{}.png".format(epoch), r_i)
                    cv2.imwrite(result_dir + "rec_img_{}.png".format(epoch), o_i)
            # train_writer.add_summary(summary1, epoch * 2)
            # train_writer.add_summary(summary2, epoch * 2 + 1)

            print()
            save_model_tf( ckpt_dir, sess, saver, epoch)
            rgb_gen_tf(sess, g_in, real_images, gen_img , epoch, do_blur=do_blur)
    else:
        if not cont:
            load_model_tf(ckpt_dir, sess, saver)
        rgb_gen_tf(sess, g_in, real_images, gen_img , 0, do_blur=do_blur)


def rgb_gan_main_keras_tf( trainit = True, cont=False, load_discrim=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    set_session(sess)
    # Generate training data
    do_blur = True
    cdim = 1
    if do_blur:
        cdim = 4
    gm  = rgb_unet(img_size, cdim = cdim)
    gm.summary()
    dm= discriminator(img_size, cdim=cdim+3)
    dm.summary()
    g_in = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, cdim])
    real_images = tf.placeholder(tf.float32, [batch_size, img_size, img_size, 3])
    gen_img = gm(g_in)
    real_AB = tf.concat([g_in, real_images], 3)
    fake_AB = tf.concat([g_in, gen_img], 3)
    disc_true_logits = dm(real_AB)
    disc_fake_logits = dm(fake_AB)


    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_true_logits, labels=tf.ones_like(disc_true_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.zeros_like(disc_fake_logits)))
    d_loss = d_loss_real + d_loss_fake
    mae_loss = tf.reduce_mean(tf.abs(real_images - gen_img))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits))) \
                  + 100.0 * mae_loss
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis_' in var.name and 'moving_mean' not in var.name and 'moving_variance' not in var.name]
    g_vars = [var for var in t_vars if 'gen_' in var.name and 'moving_mean' not in var.name and 'moving_variance' not in var.name]
    d_var_names = [var.name for var in d_vars]
    g_var_names = [var.name for var in g_vars]
    print('\n'.join(d_var_names))
    print('\n'.join(g_var_names))
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)
    sess.run(tf.global_variables_initializer() )
    if load_discrim:
        saver = tf.train.Saver()
    else:
        saver = tf.train.Saver(g_vars)
    # Train model
    #继续训练
    if cont:
        load_model_tf(ckpt_dir,sess, saver)
    all_files = os.listdir(train_data_dir)
    steps_per_epoch = 512

    rgb_gen_tf(sess, g_in, real_images, gen_img, 10000, do_blur=do_blur)
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.name, var)
    if trainit:
        imgs, _ = load_test(img_size, data_dir=test_data_dir)
        color_me, imgs, batch_edge, base_colors = get_rgb_XY(imgs, do_blur, return_edge=True)
        size = int(np.ceil(np.sqrt(len(imgs))))
        ri = utils.merge_color(imgs, (size, size)) * 255.0
        edge = utils.merge_color( batch_edge, (size, size))*255.0
        color = utils.merge_color( base_colors, (size, size))*255.0
        cv2.imwrite(result_dir + "base.png", color)
        cv2.imwrite(result_dir + "base_line.jpg", edge)
        cv2.imwrite(result_dir + "real_img.jpg", ri)
        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter('logs',
        #                                      sess.graph)
        all_batches = image_rgb_gen_batches(all_files, batch_size, img_size, do_blur=do_blur,
                                            train_data_dir=train_data_dir)
        for epoch in range( 20000 ):
            # Xtest = get_Xtrainlimit(img_size, data_dir=test_data_dir,limit=64)
            progbar = keras_generic_utils.Progbar( 100*batch_size )#len(all_files)
            batch_epoch = 100#int(len(all_files)/batch_size)
            batch_counter = 0
            for batch_counter in range( batch_epoch ):
                batch = next(all_batches)
                disc_loss, gen_mae, _ = sess.run([d_loss, mae_loss, d_optim],
                                          feed_dict={real_images: batch[1], g_in: batch[0],K.learning_phase():1 })

                gen_loss, _ = sess.run([g_loss, g_optim],
                                          feed_dict={real_images: batch[1], g_in: batch[0],K.learning_phase():1 })

                gen_total_loss = min(gen_loss+gen_mae, 10000)
                gen_mae = min(gen_mae,10000)
                gen_log_loss = min(gen_loss, 10000)
                progbar.add(batch_size, values=[("D_l", disc_loss),
                                                ("G_lt", gen_total_loss),
                                                ("G_mae", gen_mae),
                                                ("G_l", gen_log_loss)])
                if batch_counter == batch_epoch-1:
                    r_i = utils.merge_color(batch[1], (size, size)) * 255.0
                    output = sess.run(gen_img,
                                      feed_dict={real_images: batch[1], g_in: batch[0], K.learning_phase(): 0})
                    o_i = utils.merge_color(output, (size, size)) * 255.0
                    cv2.imwrite(result_dir + "real_img_{}.png".format(epoch), r_i)
                    cv2.imwrite(result_dir + "rec_img_{}.png".format(epoch), o_i)
            # train_writer.add_summary(summary1, epoch * 2)
            # train_writer.add_summary(summary2, epoch * 2 + 1)

            print()
            save_model_tf( ckpt_dir, sess, saver, epoch)
            rgb_gen_tf(sess, g_in, real_images, gen_img , epoch, do_blur=do_blur)
    else:
        if not cont:
            load_model_tf(ckpt_dir, sess, saver)
        rgb_gen_tf(sess, g_in, real_images, gen_img , 0, do_blur=do_blur)


def get_disc_batch(X_in, X_out, generator_model, batch_counter):
    label_flipping = 0
    # Create X_disc: alternatively only generated or real images
    # generate fake image
    X_disc1 = generator_model.predict(X_in)
    y_disc1 = np.zeros((X_disc1.shape[0], 1), dtype=np.float)
    y_disc1[:, 0] = 0
    # generate real image
    X_disc2 = X_out
    y_disc2 = np.zeros((X_disc2.shape[0], 1), dtype=np.float)
    y_disc2[:, 0] = 1
    X_disc1 = np.concatenate([X_disc1, X_in], axis=3)
    X_disc2 = np.concatenate([X_disc2, X_in], axis=3)
    X_disc = np.concatenate([X_disc1, X_disc2], axis=0)
    y_disc = np.concatenate([y_disc1, y_disc2], axis=0)
    if label_flipping > 0:
        p = np.random.binomial(1, label_flipping)
        if p > 0:
            y_disc[:,0] = 1- y_disc[:,0]
    return X_disc, y_disc


def load_img_test():
    do_blur = True
    imgs, _ = load_test(img_size, data_dir=test_data_dir)
    color_me, imgs, batch_edge, base_colors = get_rgb_XY(imgs, do_blur, return_edge=True)
    import utils
    size = int(np.ceil(np.sqrt(len(imgs))))
    real_img = utils.merge_color(imgs, (size, size)) * 255.0
    edge = utils.merge_color(batch_edge, (size, size)) * 255.0
    color = utils.merge_color(base_colors, (size, size)) * 255.0
    cv2.imwrite(result_dir + "base.png", color)
    cv2.imwrite(result_dir + "base_line.jpg", edge)
    cv2.imwrite(result_dir + "real_img.jpg", real_img)
    all_files = os.listdir(train_data_dir)
    bc = 0
    for batch in image_rgb_gen_batches(all_files, batch_size, img_size, do_blur=do_blur,
                                       train_data_dir=train_data_dir):
        # real_img = utils.merge_color(batch[1], (size, size)) * 255.0

        cv2.imwrite(result_dir + "real_{}.png".format(bc), batch[1][0]*255.0)
        bc+=1
        if bc>10:
            break

if __name__ == '__main__':
    if len(sys.argv)>=2:
        cmd = sys.argv[1]
        if cmd == 'gan_tf':
            rgb_gan_main(trainit=True, cont=False)
        else:
            set_keras_session()
            if cmd == 'ab':
                ab_main( trainit=True, cont=False )
            elif cmd == 'trans':
                ab_trans_main( train_it=True, cont=False )
            elif cmd == 'rgb':
                rgb_main( trainit=True, cont=False)
    else:
        # rgb_gan_main( trainit= True, cont = False)
        rgb_gan_main_keras_tf( trainit= True, cont = False)
        # load_img_test( )
        # img = utils.get_image('./data/cm.jpg',
        #                 target_size=(img_size, img_size))
        # print(img.dtype)
        # cv2.imshow('image', ((img/255.0)*255.0).astype(np.uint8) )
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print('done.')
