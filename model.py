import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.initializers import RandomNormal
import keras.backend as K

def CUNet(shape):
    
    inputs = Input(shape)
    e1 = Conv2D(64, 3, strides=(2,2), padding='same', use_bias=False)(inputs)
    e1 = LeakyReLU(alpha=0.2)(e1)
    e1 = Dropout(rate=.5)(e1)
    
    e2 = Conv2D(128, 3, strides=(2,2), padding='same', use_bias=False)(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(alpha=0.2)(e2)
    e2 = Dropout(rate=.5)(e2)
    
    e3 = Conv2D(256, 3, strides=(2,2), padding='same', use_bias=False)(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(alpha=0.2)(e3)
    e3 = Dropout(rate=.5)(e3)
    
    e4 = Conv2D(512, 3, strides=(2,2), padding='same', use_bias=False)(e3)
    e4 = BatchNormalization()(e4)
    e4 = LeakyReLU(alpha=0.2)(e4)
    e4 = Dropout(rate=.5)(e4)

    e5 = Conv2D(512, 3, strides=(2,2), padding='same', use_bias=False)(e4)
    e5 = BatchNormalization()(e5)
    e5 = LeakyReLU(alpha=0.2)(e5)
    e5 = Dropout(rate=.5)(e5)

    e6 = Conv2D(512, 3, strides=(2,2), padding='same', use_bias=False)(e5)
    e6 = BatchNormalization()(e6)
    e6 = LeakyReLU(alpha=0.2)(e6)
    e6 = Dropout(rate=.5)(e6)

    e7 = Conv2D(512, 3, strides=(2,2), padding='same', use_bias=False)(e6)
    e7 = BatchNormalization()(e7)
    e7 = LeakyReLU(alpha=0.2)(e7)
    e7 = Dropout(rate=.5)(e7)

    e8 = Conv2D(512, 3, strides=(2,2), padding='same', use_bias=False)(e7)
    e8 = BatchNormalization()(e8)
    e8 = LeakyReLU(alpha=0.2)(e8)
    
    # noise = Input((K.int_shape(e8)[1], K.int_shape(e8)[2], K.int_shape(e8)[3]))
    # m0 = Concatenate()([e8, noise])
    m0 = Dropout(rate=.5)(e8)

    d1 = Conv2DTranspose(512, 3, strides=(2,2), padding='same', use_bias=False)(m0)
    d1 = BatchNormalization()(d1)
    d1 = ReLU()(d1)
    m1 = Concatenate()([e7, d1])
    m1 = Dropout(rate=.5)(m1)

    d2 = Conv2DTranspose(512, 3, strides=(2,2), padding='same', use_bias=False)(m1)
    d2 = BatchNormalization()(d2)
    d2 = ReLU()(d2)
    m2 = Concatenate()([e6, d2])
    m2 = Dropout(rate=.5)(m2)

    d3 = Conv2DTranspose(512, 3, strides=(2,2), padding='same', use_bias=False)(m2)
    d3 = BatchNormalization()(d3)
    d3 = ReLU()(d3)
    m3 = Concatenate()([e5, d3])
    m3 = Dropout(rate=.5)(m3)

    d4 = Conv2DTranspose(512, 3, strides=(2,2), padding='same', use_bias=False)(m3)
    d4 = BatchNormalization()(d4)
    d4 = ReLU()(d4)
    m4 = Concatenate()([e4, d4])
    m4 = Dropout(rate=.5)(m4)

    d5 = Conv2DTranspose(256, 3, strides=(2,2), padding='same', use_bias=False)(m4)
    d5 = BatchNormalization()(d5)
    d5 = ReLU()(d5)
    m5 = Concatenate()([e3, d5])
    m5 = Dropout(rate=.5)(m5)

    d6 = Conv2DTranspose(128, 3, strides=(2,2), padding='same', use_bias=False)(m5)
    d6 = BatchNormalization()(d6)
    d6 = ReLU()(d6)
    m6 = Concatenate()([e2, d6])
    m6 = Dropout(rate=.5)(m6)

    d7 = Conv2DTranspose(64, 3, strides=(2,2), padding='same', use_bias=False)(m6)
    d7 = BatchNormalization()(d7)
    d7 = ReLU()(d7)
    m7 = Concatenate()([e1, d7])
    m7 = Dropout(rate=.5)(m7)

    d8 = Conv2DTranspose(64, 3, strides=(2,2), padding='same', use_bias=False)(m7)
    d8 = BatchNormalization()(d8)
    d8 = ReLU()(d8)
    d8 = Dropout(rate=.5)(d8)

    output = Conv2D(1, 1, activation='tanh')(d8)
    model = tf.keras.Model(inputs=[inputs], outputs=output)
    model.summary()
    return model


def UNet(shape):
    inputs = Input(shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    noise = Input((K.int_shape(conv5)[1], K.int_shape(conv5)[2], K.int_shape(conv5)[3]))
    conv5 = Concatenate()([conv5, noise])

    up6 = Conv2D(
        512,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(
        256,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(
        128,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='relu',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))
    up9 = ZeroPadding2D(((0, 1), (0, 1)))(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
    conv10 = Conv2D(1, 1, activation='tanh')(conv9)

    model = tf.keras.Model(inputs=[inputs, noise], outputs=conv10)
    model.summary()
    return model


def patch_discriminator(shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=shape)
    cond_image = Input((256, 256, 4))
    conc_img = Concatenate()([in_image, cond_image])
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(conc_img)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    x = LeakyReLU(alpha=0.2)(d)
    output = Conv2D(1, (4, 4), padding='same',
                    activation='sigmoid', kernel_initializer=init)(d)
    model = Model([in_image, cond_image], output)

    return model


def mount_discriminator_generator(g, d, image_shape):
    d.trainable = False
    input_gen = Input(shape=image_shape)
    input_noise = Input(shape=(16, 16, 512))
    gen_out = g([input_gen])
    output_d = d([gen_out, input_gen])
    model = Model(inputs=[input_gen], outputs=[output_d, gen_out])
    model.summary()

    return model
