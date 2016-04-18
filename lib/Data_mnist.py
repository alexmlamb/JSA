from fuel.datasets.youtube_audio import YouTubeAudio
import random

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

import numpy as np

import scipy

import gzip

import cPickle as pickle

class Data:

    def __init__(self, mb_size):
        self.mb_size = mb_size

        file_loc = "/u/lambalex/data/mnist/mnist.pkl.gz"

        f = gzip.open(file_loc, 'rb')

        self.train_set, self.valid_set, self.test_set = pickle.load(f)

        self.train_X = self.train_set[0]

    '''
        Pick a random location, get sequence of seq_length
    '''
    def getBatch(self):
        startingPoint = random.randint(0, self.train_X.shape[0] - self.mb_size - 1)
        return self.train_X[startingPoint : startingPoint + self.mb_size,:].reshape(self.mb_size, 1, 28, 28)

    def saveExample(self, x_gen, name):

        assert x_gen.ndim == 3

        x_gen = np.clip(x_gen, 0.01, 0.99)

        imgLoc = "plots/" + name + ".png"

        scipy.misc.imsave(imgLoc, x_gen[0])

if __name__ == "__main__":
    d = Data(mb_size = 2)

    x = d.getBatch()

    print x.shape

    d.saveExample(x[0], name = 'derp')

    print x.tolist()

