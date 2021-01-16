from os import PRIO_PGRP
from model import *
import numpy as np
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model
import time
#from dataset_builder import *


def generate_real_samples(dataset, ground_trud_ds, n_samples, patch_size):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    gt = ground_trud_ds[ix]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return X, y, gt


def generate_fake_samples(g_model, dataset, patch_size):
    w_noise = np.random.normal(0, 1, (dataset.shape[0], 16, 16, 512))
    X = g_model.predict([dataset])
    y = np.zeros((len(X), patch_size, patch_size, 1))
    return X, y


def sample_images(generator, source, target, idx, name = ""):

    target = np.uint8(target * 127.5 + 127.5)
    w_noise = np.random.normal(0, 1, (1, 16, 16, 512))
    print(source.shape)
    predicted = generator.predict([source])
    im = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    im_source = np.uint8(source[0, ...] * 255)
    
    # im_c = np.concatenate((np.squeeze(im, axis=-1),
    #                        np.squeeze(target, axis=-1),
    #                        np.squeeze(im_source, axis=-1)), axis=1)
    im_c = np.concatenate((np.squeeze(im, axis=-1), np.squeeze(target, axis=-1),
        im_source[..., 0], im_source[..., 1], im_source[..., 2], im_source[..., 3]), axis=1)
    plt.imsave('./outputs/sketch_conversion' + str(idx) + name + '.png', im_c, cmap='terrain')
    

def test_gan():
    generator = 'generatorD1x1'
    terrain_generator = load_model(generator + '.h5')
    data = np.load('training_data_3.npz')
    XTrain = data['x']
    YTrain = data['y']
    for i in range(500):
        source = XTrain[i:i + 1, ...]
        target = YTrain[i, ...]
        sample_images(terrain_generator, source, target, i, generator)

def load_train_config():
    return load_model('terrain_generator_41x1D.h5')


def process_img(data, terrain_generator):
    w_noise = np.random.normal(0, 1, (1, 16, 16, 512))
    image = terrain_generator.predict([data])
    im = np.uint8(image[0, ...] * 127.5 + 127.5)
    im = np.squeeze(im, axis=-1)
    plt.imsave("./IHopeThisWorks.png", im, cmap="terrain")
    plt.imsave("./heightmap.png", 1 - im, cmap="Greys")


def train_gan():
    start_time = time.perf_counter()
    data = np.load('training_data_4.npz')
    XTrain = data['x']
    YTrain = data['y']
    input_shape_gen = (XTrain.shape[1], XTrain.shape[2], XTrain.shape[3])
    input_shape_disc = (YTrain.shape[1], YTrain.shape[2], YTrain.shape[3])
    terrain_generator = CUNet(input_shape_gen)
    terrain_discriminator = patch_discriminator(input_shape_disc)
    optd = Adam(0.0001, 0.5)
    terrain_discriminator.compile(loss='binary_crossentropy', optimizer=optd)
    composite_model = mount_discriminator_generator(
        terrain_generator, terrain_discriminator, input_shape_gen)
    composite_model.compile(
        loss=[
            'binary_crossentropy', 'mae'], loss_weights=[
            1, 3], optimizer=optd)

    n_epochs, n_batch, = 200, 20
    bat_per_epo = int(len(XTrain) / n_batch)
    patch_size = 16
    n_steps = bat_per_epo * n_epochs + 1
    min_loss = 999
    avg_loss = 0
    avg_dloss = 0
    for i in range(n_steps):
        X_real, labels_real, Y_target = generate_real_samples(XTrain, YTrain, n_batch, patch_size)
        Y_target[np.isnan(Y_target)] = 0
        X_real[np.isnan(X_real)] = 0
        Y_fake, labels_fake = generate_fake_samples(terrain_generator, X_real, patch_size)
        w_noise = np.random.normal(0, 1, (n_batch, 16, 16, 512))
        losses_composite = composite_model.train_on_batch(
            [X_real], [labels_real, Y_target])

        loss_discriminator_fake = terrain_discriminator.train_on_batch(
            [Y_fake, X_real], labels_fake)
        loss_discriminator_real = terrain_discriminator.train_on_batch(
            [Y_target, X_real], labels_real)
        d_loss = (loss_discriminator_fake + loss_discriminator_real) / 2
        avg_dloss = avg_dloss + (d_loss - avg_dloss) / (i + 1)
        avg_loss = avg_loss + (losses_composite[0] - avg_loss) / (i + 1)
        current_time = time.perf_counter()
        print('step ' + str(i) + ' of ' + str(n_steps) +
              ' total loss:' + str(avg_loss) + ' d_loss:' + str(avg_dloss) +
              ' current execution time: ' + str(round(current_time - start_time, 2)))

        if i % 100 == 0:
            sample_images(terrain_generator, X_real[0:1, ...], Y_target[0, ...], i)
        if i % 500 == 0:
            terrain_discriminator.save('terrain_discriminator_41x1D' + '.h5', True)
            terrain_generator.save('terrain_generator_41x1D' + '.h5', True)

if __name__ == '__main__':
    # extract_patches_from_raster()
    # compute_sketches()
    # train_gan()
    # test_gan()
    print("load")