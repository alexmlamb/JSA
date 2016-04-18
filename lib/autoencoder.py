from Data_mnist import Data
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer
from HiddenLayer import HiddenLayer
from ConvolutionalLayer import ConvPoolLayer
import time
from DeConvLayer import DeConvLayer

from consider_constant import consider_constant

from latent_discriminator import init_params_ldisc, latent_discriminator

def init_params_encoder(config):
    params = {}

    scale = 0.05
    num_latent = config['num_latent']

    num_hidden = config['num_hidden']
    
    #Conv:
    #In, Kernel, Kernel, Out

    #Deconv:
    #In, Out, Kernel, Kernel

    '''
    Mnist starts as 28 x 28 x 3

    14 x 14 x 96
    7 x 7 x 128
    3 x 3 x 256

    '''

    #params["Wc_enc_1"] = theano.shared(scale * np.random.normal(size = (3, 5, 5, 96)).astype('float32'))
    #params["bc_enc_1"] = theano.shared(scale * np.random.normal(size = (96)).astype('float32'))

    #params["Wc_enc_2"] = theano.shared(scale * np.random.normal(size = (96, 3, 3, 128)).astype('float32'))
    #params["bc_enc_2"] = theano.shared(scale * np.random.normal(size = (128)).astype('float32'))

    #params["Wc_enc_3"] = theano.shared(scale * np.random.normal(size = (128, 3, 3, 256)).astype('float32'))
    #params["bc_enc_3"] = theano.shared(scale * np.random.normal(size = (256)).astype('float32'))

    params["W_enc_1"] = theano.shared(scale * np.random.normal(size = (784, num_hidden)).astype('float32'))
    params["W_enc_2"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))

    params["b_enc_1"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params["b_enc_2"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))

    params["z_mean_W"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_latent)).astype('float32'))
    params["z_mean_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    params["z_std_W"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_latent)).astype('float32'))
    params["z_std_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    return params

def init_params_decoder(config):

    params = {}

    scale = 0.05
    num_latent = config['num_latent']
    num_hidden = config['num_hidden']

    params['W_dec_1'] = theano.shared(scale * np.random.normal(size = (num_latent * 2, num_hidden)).astype('float32'))
    params['W_dec_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_dec_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, 784)).astype('float32'))

    params['b_dec_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_3'] = theano.shared(0.0 * np.random.normal(size = (784)).astype('float32'))

    #params['Wc_dec_1'] = theano.shared(scale * np.random.normal(size = (512, 256, 5,5)).astype('float32'))
    #params['bc_dec_1'] = theano.shared(0.0 * np.random.normal(size = (256)).astype('float32'))

    #params['Wc_dec_2'] = theano.shared(scale * np.random.normal(size = (256, 128, 5,5)).astype('float32'))
    #params['bc_dec_2'] = theano.shared(0.0 * np.random.normal(size = (128)).astype('float32'))

    #params['Wc_dec_3'] = theano.shared(scale * np.random.normal(size = (128, 1, 5,5)).astype('float32'))
    #params['bc_dec_3'] = theano.shared(0.0 * np.random.normal(size = (1)).astype('float32'))


    return params

def init_params_disc(config):

    params = {}

    scale = 0.05
    num_hidden = config['num_hidden']
    num_latent = config['num_latent']

    #in x kernel x kernel x out

    #params["W_c_1"] = theano.shared(scale * np.random.normal(size = (3, 5, 5, 128)).astype('float32'))
    #params["b_c_1"] = theano.shared(scale * np.random.normal(size = (128)).astype('float32'))

    #params["W_c_2"] = theano.shared(scale * np.random.normal(size = (128, 3, 3, 256)).astype('float32'))
    #params["b_c_2"] = theano.shared(scale * np.random.normal(size = (256)).astype('float32'))

    #params["W_c_3"] = theano.shared(scale * np.random.normal(size = (256, 3, 3, 512)).astype('float32'))
    #params["b_c_3"] = theano.shared(scale * np.random.normal(size = (512)).astype('float32'))

    #params["W_ch_1"] = theano.shared(scale * np.random.normal(size = (512 * 6 * 6, num_hidden)).astype('float32'))
    #params["b_ch_1"] = theano.shared(scale * np.random.normal(size = (num_hidden)).astype('float32'))

    params['W_disc_1'] = theano.shared(scale * np.random.normal(size = (784 + num_latent, num_hidden)).astype('float32'))
    params['W_disc_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_disc_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_disc_4'] = theano.shared(scale * np.random.normal(size = (num_hidden, 1)).astype('float32'))

    params['b_disc_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_3'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_4'] = theano.shared(0.0 * np.random.normal(size = (1)).astype('float32'))

    print "DEFINED ALL DISCRIMINATOR WEIGHTS"

    return params

def normalize(x):
    return x

def denormalize(x):
    return x

import random as rng
srng = theano.tensor.shared_randomstreams.RandomStreams(420)

def dropout(in_layer, p = 0.5):
    return in_layer * T.cast(srng.binomial(n=1,p=p,size=in_layer.shape),'float32')

def noise(in_layer, eta = 0.1):
    return in_layer + eta * srng.normal(size=in_layer.shape)

'''
Takes real samples and generated samples.  

Two fully connected layers, then a classifier output.  

Takes x_real, x_reconstructed, and z (reconstruction)

Feed both (x_real, z) and (x_reconstructed, z) to the discriminator.  

Run three times: D_real, D_fake, G_fake.  When we run with G_fake, pass in consider_constant(z).  

'''
def discriminator(x, z, params, mb_size, num_hidden, num_latent):

    import random as rng
    srng = theano.tensor.shared_randomstreams.RandomStreams(420)

    #c_1 = ConvPoolLayer(in_length = 4000, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_1'], b = params['b_c_1'])

    #c_2 = ConvPoolLayer(in_length = 399, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_2'], b = params['b_c_2'])

    #c_3 = ConvPoolLayer(in_length = 38, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_3'], b = params['b_c_3'])

    #c_h_1 = HiddenLayer(num_in = 6 * 512, num_out = num_hidden, W = params['W_ch_1'], b = params['b_ch_1'], activation = 'relu', batch_norm = False)

    h_out_1 = HiddenLayer(num_in = num_hidden + num_latent, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_disc_1'], b = params['b_disc_1'])

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_disc_2'], b = params['b_disc_2'])

    h_out_3 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_disc_3'], b = params['b_disc_3'])

    h_out_4 = HiddenLayer(num_in = num_hidden, num_out = 1, activation = None, batch_norm = False, W = params['W_disc_4'], b = params['b_disc_4'])

    #c_1_value = c_1.output(dropout(x, 0.8))

    #c_2_value = c_2.output(c_1_value)

    #c_3_value = c_3.output(c_2_value)

    #c_h_1_value = c_h_1.output(c_3_value.flatten(2))

    h_out_1_value = dropout(h_out_1.output(T.concatenate([z, dropout(noise(x.flatten(2)), 0.8)], axis = 1)), 0.5)

    h_out_2_value = dropout(h_out_2.output(h_out_1_value), 0.5)

    h_out_3_value = dropout(h_out_3.output(h_out_2_value), 0.5)

    h_out_4_value = h_out_4.output(h_out_3_value)

    raw_y = h_out_4_value

    classification = T.nnet.sigmoid(raw_y)

    results = {'c' : classification}

    return results

'''
Maps from a given x to an h_value.  


'''
def encoder(x, params, config):

    mb_size = config['mb_size']
    num_hidden = config['num_hidden']

    x = T.specify_shape(x, (128, 1, 28, 28))

    #c_1 = ConvPoolLayer(in_length = 4000, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = True, W = params['Wc_enc_1'], b = params['bc_enc_1'])

    #c_2 = ConvPoolLayer(in_length = 399, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = True, W = params['Wc_enc_2'], b = params['bc_enc_2'])

    #c_3 = ConvPoolLayer(in_length = 38, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = True, W = params['Wc_enc_3'], b = params['bc_enc_3'])

    h_out_1 = HiddenLayer(num_in = 784, num_out = num_hidden, W = params['W_enc_1'], b = params['b_enc_1'], activation = 'relu', batch_norm = True)

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, W = params['W_enc_2'], b = params['b_enc_2'], activation = 'relu', batch_norm = True)

    print "x ndim", x.ndim

    #c_1_value = T.specify_shape(c_1.output(x), (128, 96, 16, 16))
    #c_2_value = c_2.output(c_1_value)
    #c_3_value = c_3.output(c_2_value)

    h_out_1_value = T.specify_shape(h_out_1.output(x.flatten(2)), (128, num_hidden))
    h_out_2_value = h_out_2.output(h_out_1_value)

    return {'h' : h_out_2_value}

'''
Maps from a given z to a decoded x.  

'''
def decoder(z, z_extra, params, config):

    mb_size = config['mb_size']
    num_latent = config['num_latent']
    num_hidden = config['num_hidden']

    h_out_1 = HiddenLayer(num_in = num_latent, num_out = num_hidden, W = params['W_dec_1'], b = params['b_dec_1'], activation = 'relu', batch_norm = True)

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, W = params['W_dec_2'], b = params['b_dec_2'], activation = 'relu', batch_norm = True)

    h_out_3 = HiddenLayer(num_in = num_hidden, num_out = 784, activation = None, W = params['W_dec_3'], b = params['b_dec_3'], batch_norm = False)

    #c1 = DeConvLayer(in_channels = 512, out_channels = 256, activation = 'relu', W = params['Wc_dec_1'], b = params['bc_dec_1'], batch_norm = True)

    #c2 = DeConvLayer(in_channels = 256, out_channels = 128, activation = 'relu', W = params['Wc_dec_2'], b = params['bc_dec_2'], batch_norm = False)

    #c3 = DeConvLayer(in_channels = 128, out_channels = 1, activation = None, W = params['Wc_dec_3'], b = params['bc_dec_3'], batch_norm = False)

    z = T.concatenate([z,z_extra], axis = 1)

    h_out_1_value = h_out_1.output(z)
    h_out_2_value = h_out_2.output(h_out_1_value)
    h_out_3_value = h_out_3.output(h_out_2_value)

    #c1_o = c1.output(h_out_3_value.reshape((128,512,8,1)))
    #c2_o = c2.output(c1_o)
    #c3_o = c3.output(c2_o)

    out = h_out_3_value.reshape((128, 1, 28, 28))

    return {'h' : out}

'''
Given x (unormalized), returns a reconstructed_x and a sampled x (both unnormalized)
'''

def define_network(x, params, config):

    num_hidden = config['num_hidden']
    mb_size = config['mb_size']
    num_latent = config['num_latent']

    enc = encoder(x, params, config)

    mean_layer = DenseLayer((mb_size, num_hidden), num_units = num_latent, nonlinearity=None, W = params['z_mean_W'], b = params['z_mean_b'])
    std_layer = DenseLayer((mb_size, num_hidden), num_units = num_latent, nonlinearity=None, W = params['z_std_W'], b = params['z_std_b'])

    mean = mean_layer.get_output_for(enc['h'])
    std = T.exp(std_layer.get_output_for(enc['h']))

    import random as rng
    srng = theano.tensor.shared_randomstreams.RandomStreams(420)

    z_sampled = srng.normal(size = mean.shape, avg = 0.0, std = 1.0)
    z_extra = 0.0 * srng.normal(size = mean.shape, avg = 0.0, std = 1.0)

    z_reconstruction = mean + (0.0 + std * 0.0) * srng.normal(size = mean.shape, avg = 0.0, std = 1.0)

    #z_var = std**2
    z_loss = 0.0 * 0.5 * T.sum(T.clip(mean**2, 4.0, 999999.9) + std**2 - T.log(std**2) - 1.0)

    dec_reconstruction = decoder(z_reconstruction, z_extra, params, config)
    dec_sampled = decoder(z_sampled, z_extra, params, config)

    interp_lst = []

    for j in range(0,128):
        interp_lst.append(z_reconstruction[0] * (j/128.0) + z_reconstruction[-1] * (1 - j / 128.0))

    z_interp = T.concatenate([interp_lst], axis = 1)

    dec_interp = decoder(z_interp, z_extra, params, config)

    results_map = {'reconstruction' : dec_reconstruction['h'], 'z_loss' : z_loss, 'sample' : dec_sampled['h'], 'interp' : dec_interp['h'], 'z' : z_reconstruction}

    return results_map

def compute_loss(x, x_reconstructed):
    return 100.0 * T.mean(T.sqr((x - x_reconstructed)))# + T.abs_(T.mean(x) - T.mean(x_reconstructed)) + T.mean(T.abs_(T.std(x) - T.std(x_reconstructed)))

if __name__ == "__main__":

    config = {}
    config['mb_size'] = 128
    config['num_hidden'] = 2048
    config['num_latent'] = 2048

    d = Data(mb_size = config['mb_size'])

    #todo make sure disc is only updating on disc_params.  

    params_enc = init_params_encoder(config)
    params_dec = init_params_decoder(config)
    params_disc = init_params_disc(config)
    params_ldisc = init_params_ldisc(config)

    params = {}
    params.update(params_enc)
    params.update(params_dec)

    for pv in params_enc.values() + params_disc.values() + params_dec.values():
        print pv.dtype

    x = T.tensor4()

    results_map = define_network(normalize(x), params, config)

    x_reconstructed = results_map['reconstruction']
    x_sampled = results_map['sample']

    

    ldisc = latent_discriminator(T.concatenate([], axis = 0), params_ldisc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])

    disc_real_D = discriminator(normalize(x), results_map['z'], params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])
    disc_fake_D = discriminator(x_reconstructed, results_map['z'], params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])
    disc_fake_G = discriminator(x_reconstructed, consider_constant(results_map['z']), params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])

    bce = T.nnet.binary_crossentropy

    LD_dD = bce(disc_real_D['c'], 0.999 * T.ones(disc_real_D['c'].shape)).mean() + bce(disc_fake_D['c'], 0.0001 + T.zeros(disc_fake_D['c'].shape)).mean()
    LD_dG = bce(disc_fake_G['c'], T.ones(disc_fake_G['c'].shape)).mean()

    vae_loss = results_map['z_loss']
    rec_loss = compute_loss(x_reconstructed, normalize(x))

    loss = vae_loss + rec_loss

    inputs = [x]

    outputs = {'loss' : loss, 'vae_loss' : vae_loss, 'rec_loss' : rec_loss, 'reconstruction' : denormalize(x_reconstructed), 'c_real' : disc_real_D['c'], 'c_fake' : disc_fake_D['c'], 'x' : x, 'sample' : denormalize(x_sampled), 'interp' : denormalize(results_map['interp']), 'ldisc' : }

    print "params", params.keys()
    print "params enc", params_enc.keys()
    print "params dec", params_dec.keys()
    print "params_disc", params_disc.keys()

    updates = lasagne.updates.adam(LD_dG, params_dec.values(), learning_rate = 0.001, beta1 = 0.5)
    updates_disc = lasagne.updates.adam(LD_dD + vae_loss, params_disc.values() + params_enc.values(), learning_rate = 0.0001, beta1 = 0.5)
    updates_ldisc = lasagne.updates.adam(ldisc_dD, params_ldisc.values(), learning_rate = 0.0001, beta1 = 0.5)

    train_method = theano.function(inputs = inputs, outputs = outputs, updates = updates)
    disc_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_disc)
    ldisc_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_ldisc)
    #gen_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_gen)

    last_acc = 0.0
    score_diff = 0.0

    for i in range(0,10000000):
        x = d.getBatch()

        t0 = time.time()
        res = train_method(x)
        disc_method(x)

        if i % 20 == 1:
            print "time", time.time() - t0

        if i % 200 == 1:
            d.saveExample(res['reconstruction'][0], "image_reconstruction")
            d.saveExample(x[0][:200], "image_original")
            d.saveExample(res['sample'][0],'image_sample')


        if i % 20 == 1:
            print "==========================================================="
            print ""

            print "update", i, "loss", res['loss']
            print "vae loss", res['vae_loss']
            print "rec loss", res['rec_loss']
    
            print "classification", res['c_real'][:20].tolist()
            print res['c_fake'][:20].tolist()
            print "real", res['c_real'].mean(), "fake", res['c_fake'].mean()
            score_diff = res['c_real'].mean() - res['c_fake'].mean()
            #print res['classification'].tolist()




