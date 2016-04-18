

def init_params_latent_disc(config):

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

    params['W_ldisc_1'] = theano.shared(scale * np.random.normal(size = (784 + num_latent, num_hidden)).astype('float32'))
    params['W_ldisc_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_ldisc_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_ldisc_4'] = theano.shared(scale * np.random.normal(size = (num_hidden, 1)).astype('float32'))

    params['b_ldisc_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_ldisc_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_ldisc_3'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_ldisc_4'] = theano.shared(0.0 * np.random.normal(size = (1)).astype('float32'))

    print "DEFINED ALL LATENT DISCRIMINATOR WEIGHTS"

    return params

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
def latent_discriminator(x, z, params, mb_size, num_hidden, num_latent):

    import random as rng
    srng = theano.tensor.shared_randomstreams.RandomStreams(420)

    #c_1 = ConvPoolLayer(in_length = 4000, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_1'], b = params['b_c_1'])

    #c_2 = ConvPoolLayer(in_length = 399, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_2'], b = params['b_c_2'])

    #c_3 = ConvPoolLayer(in_length = 38, batch_size = mb_size, stride = 2, activation = "relu", batch_norm = False, W = params['W_c_3'], b = params['b_c_3'])

    #c_h_1 = HiddenLayer(num_in = 6 * 512, num_out = num_hidden, W = params['W_ch_1'], b = params['b_ch_1'], activation = 'relu', batch_norm = False)

    h_out_1 = HiddenLayer(num_in = num_hidden + num_latent, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_ldisc_1'], b = params['b_ldisc_1'])

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_ldisc_2'], b = params['b_ldisc_2'])

    h_out_3 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, activation = 'relu', batch_norm = False, W = params['W_ldisc_3'], b = params['b_ldisc_3'])

    h_out_4 = HiddenLayer(num_in = num_hidden, num_out = 1, activation = None, batch_norm = False, W = params['W_ldisc_4'], b = params['b_ldisc_4'])

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









