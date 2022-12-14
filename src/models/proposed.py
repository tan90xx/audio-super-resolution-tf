import numpy as np
import tensorflow as tf

from scipy import interpolate
from .model import Model, default_opt

from .layers.subpixel import SubPixel1D, SubPixel1D_v2

from tensorflow.python.keras import backend as K
from keras.layers import add
from keras.layers.core import Activation, Dropout
from keras.layers import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import RandomNormal, Orthogonal, Constant

# ----------------------------------------------------------------------------
# Wrong:
# Conv1D init-->kernel_initializer
#        subsample_length-->strides
# Convolution1D-->Conv1D
# stdev-->stddev
# merge-->add
# x = PReLU(alpha_initializer=Constant(value=0.2),shared_axes=[1, 2])(x)-->Conv1d

class Proposed(Model):
    """Generic tensorflow model training code"""

    def __init__(self, from_ckpt=False, n_dim=None, r=2,
                 opt_params=default_opt, log_prefix='./run'):
        # perform the usual initialization
        self.r = r
        Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                       opt_params=opt_params, log_prefix=log_prefix)

    def create_model(self, n_dim, r):
        # load inputs
        X, _, _ = self.inputs
        K.set_session(self.sess)

        with tf.compat.v1.name_scope('generator'):
            x = X
            L = self.layers
            # dim/layer: 4096, 2048, 1024, 512, 256, 128,  64,  32,
            #n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
            #n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
            n_filters = [64, 64, 64, 128, 128, 128, 256, 256]
            n_filtersizes = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
            downsampling_l = []

            print('building model...')

            # downsampling layers
            for l, nf, fs in zip(list(range(L)), n_filters, n_filtersizes):
                with tf.compat.v1.name_scope('downsc_conv%d' % l):
                    x = (Conv1D(filters=nf, kernel_size=fs,
                            activation=PReLU(alpha_initializer=Constant(value=0.2),shared_axes=[1, 2]),
                            padding='same', kernel_initializer=Orthogonal(),
                            strides=2))(x)
                    #################################
                    if l%3 == 2:
                        x = Dropout(rate=0.2)(x)
                        print('Dropout')
                    print('D-Block: ', x.get_shape())
                    downsampling_l.append(x)

            # bottleneck layer
            with tf.compat.v1.name_scope('bottleneck_conv'):
                x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1],
                        activation=PReLU(alpha_initializer=Constant(value=0.2),shared_axes=[1, 2]),
                        padding='same', kernel_initializer=Orthogonal(),
                        strides=2))(x)
                ################################
                x = Dropout(rate=0.2)(x)
                print('Dropout')
                print('B-Block: ', x.get_shape())

            # upsampling layers
            for l, nf, fs, l_in in reversed(list(zip(list(range(L)), n_filters, n_filtersizes, downsampling_l))):
                with tf.compat.v1.name_scope('upsc_conv%d' % l):
                    # (-1, n/2, 2f)
                    x = (Conv1D(filters=2*nf, kernel_size=fs,
                            activation=PReLU(alpha_initializer=Constant(value=0.0),shared_axes=[1, 2]), 
                            padding='same', kernel_initializer=Orthogonal()))(x)
                    # (-1, n, f)
                    x = SubPixel1D(x, r=2)
                    # (-1, n, 2f)
                    ###################################
                    if l%3 == 2:
                        x = Dropout(rate=0.2)(x)
                        print('Dropout')
                    x = K.concatenate(tensors=[x, l_in], axis=2)
                    print('U-Block: ', x.get_shape())

            # final conv layer
            with tf.compat.v1.name_scope('lastconv'):
                x = Conv1D(filters=2, kernel_size=9,
                        activation='linear', padding='same', kernel_initializer=RandomNormal(stddev=1e-3))(x)
                x = SubPixel1D(x, r=2)
                print(x.get_shape())

            g = add([x,X])

        return g

    def predict(self, X):
        print("predicting")
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
        X = x_sp.reshape((1,len(x_sp),1))
        print((X.shape))
        feed_dict = self.load_batch((X,X), train=False)
        return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def spline_up(x_lr, r):
    x_lr = x_lr.flatten()
    x_hr_len = len(x_lr) * r
    x_sp = np.zeros(x_hr_len)

    i_lr = np.arange(x_hr_len, step=r)
    i_hr = np.arange(x_hr_len)

    f = interpolate.splrep(i_lr, x_lr)

    x_sp = interpolate.splev(i_hr, f)

    return x_sp
