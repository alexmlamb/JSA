import sys

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights



class ConvPoolLayer(object):

    def __init__(self, W, b, batch_size, in_length, stride = 1, activation = "relu", batch_norm = False, unflatten_input = None):

        W_shape = W.get_value().shape

        in_channels = W_shape[0]
        kernel_len = W_shape[1]
        out_channels = W_shape[3]

        if kernel_len == 1:
            self.padsize = 0
        elif kernel_len == 3:
            self.padsize = 1
        elif kernel_len == 5:
            self.padsize = 2
        elif kernel_len == 7:
            self.padsize = 3
        elif kernel_len == 11:
            self.padsize = 5
        else:
            raise Exception()
        self.batch_norm = batch_norm
        bias_init = 0.0
        self.activation = activation

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_length = in_length
        self.batch_size = batch_size
        self.stride = stride

        self.W = W
        self.b = b

        if batch_norm:
            self.bn_mean = theano.shared(np.zeros(shape = (1,out_channels,1,1)).astype('float32'))
            self.bn_std = theano.shared(np.random.normal(1.0, 0.001, size = (1,out_channels,1,1)).astype('float32'))


    def output(self, input):

        W_shuffled = self.W.dimshuffle(3, 0, 1, 2)  # c01b to bc01

        print "input ndim", input.ndim

        conv_out = dnn.dnn_conv(img=input,
                                        kerns=W_shuffled,
                                        subsample=(self.stride, self.stride),
                                        border_mode=self.padsize)

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if self.batch_norm:
            conv_out = (conv_out - T.mean(conv_out, axis = (0,2,3), keepdims = True)) / (1.0 + T.std(conv_out, axis=(0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(self.bn_std,0,2,3) + T.addbroadcast(self.bn_mean, 0,2,3)

        self.out_store = conv_out

        if self.activation == "relu":
            self.out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            self.out = T.tanh(conv_out)
        elif self.activation == None:
            self.out = conv_out


        return T.specify_shape(self.out, (self.batch_size, self.out_channels, self.in_length / self.stride, self.in_length / self.stride))

