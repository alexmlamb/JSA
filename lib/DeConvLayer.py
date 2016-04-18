
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


class DeConvLayer(object):

    def __init__(self, in_channels, out_channels, activation, W, b,batch_norm = False):

        #self.filter_shape = np.asarray((in_channels, out_channels, kernel_len, kernel_len))

        kernel_len = W.get_value().shape[2]
        print "kernel len", kernel_len

        self.activation = activation


        self.batch_norm = batch_norm

        self.W = W
        self.b = b

    def output(self, input):

        conv_out = deconv(input, self.W, subsample=(2,2), border_mode=(2,2))


        if self.batch_norm:
            conv_out = (conv_out - conv_out.mean(axis = (0,2,3), keepdims = True)) / (1.0 + conv_out.std(axis = (0,2,3), keepdims = True))

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if self.activation == "relu":
            out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            out = T.tanh(conv_out)
        elif self.activation == None:
            out = conv_out
        else:
            raise Exception()


        return out






