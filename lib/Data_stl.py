from fuel.datasets.youtube_audio import YouTubeAudio
import random

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import scipy

import gzip

import cPickle as pickle

from plot import plot_image_grid

import scipy.io

import h5py

from matplotlib import cm, pyplot

class Data:

    def __init__(self, mb_size):
        self.mb_size = mb_size

        file_loc_unlabeled = "/u/lambalex/data/stl/unlabeled.mat"

        UX = h5py.File(file_loc_unlabeled)['X'].value.T

        #self.train_set, self.valid_set, self.test_set = pickle.load(f)

        #self.train_X = self.train_set[0]

        #np.transpose(np.swapaxes(train_X,1,0).reshape((100000, 3, 96, 96)), (0,3,2,1))

        self.train_X = UX.reshape((100000, 3, 96, 96)).transpose(0,1,3,2)


    '''
        Pick a random location, get sequence of seq_length
    '''
    def getBatch(self):
        startingPoint = random.randint(0, self.train_X.shape[0] - self.mb_size - 1)
        return self.train_X[startingPoint : startingPoint + self.mb_size]


    def saveExample(self, x_gen, name):

        assert x_gen.ndim == 4

        x_gen = np.clip(x_gen, 0.0, 255.0)

        print x_gen.min(), x_gen.max()

        imgLoc = "plots/" + name + ".png"

        #scipy.misc.imsave(imgLoc, x_gen[0])

        images = []

        for i in range(0, x_gen.shape[0]):
            images.append(x_gen[i])

        plot_image_grid(images[:3*6], 3, 6, imgLoc)

if __name__ == "__main__":
    d = Data(mb_size = 128)

    x = d.getBatch()

    print x.shape

    d.saveExample(x, name = 'derp')


