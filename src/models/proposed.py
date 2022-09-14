import numpy as np
import tensorflow as tf

from scipy import interpolate
from .model import Model, default_opt

from .layers.subpixel import SubPixel1D, SubPixel1D_v2

from tensorflow.python.keras import backend as K
from keras.layers import add
from keras.layers.core import Activation, Dropout
from keras.layers import Conv1D
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import RandomNormal, Orthogonal

# ttyadd:
# import soundfile as sf
# from .models_keras import Conv1D
# ----------------------------------------------------------------------------
# Wrong:
# Conv1D init-->kernel_initializer
#        subsample_length-->strides
# Convolution1D-->Conv1D
# stdev-->stddev
# merge-->add
# X shape (batch, frame, length, channels)
# LeakyReLU(0.2)(x)-->PReLU(shared_axes=[1, 2])(x)
# ----------------------------------------------------------------------------
# Proposed in paper Towards Rubst Speech Super-Resolution, adopted from audiounet
def _prelu(_x, name):
  _alpha = tf.compat.v1.get_variable(name + "prelu",
              shape = _x.get_shape()[-1],
              dtype = _x.dtype,
              initializer = tf.constant_initializer(0.1))
  pos = tf.nn.relu(_x)
  neg = _alpha * (_x - tf.abs(_x)) * 0.5
  return pos + neg

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
            # ttyadd: parameters mentioned in paper
            n_filters = [64, 64, 64, 128, 128, 128, 256, 256]
            n_filtersizes = [11, 11, 11, 11, 11, 11, 11, 11, 11]
            downsampling_l = []

            print('building model...')

            # downsampling layers
            for l, nf, fs in zip(list(range(L)), n_filters, n_filtersizes):
                with tf.compat.v1.name_scope('downsc_conv%d' % l):
                    x = (Conv1D(filters=nf, kernel_size=fs,
                            activation=None, padding='same', kernel_initializer=Orthogonal(),
                            strides=2))(x)
                    #if l > 0: x = BatchNormalization(mode=2)(x)
                    # ttyadd: prelu mentioned in paper
                    #x = LeakyReLU(0.2)(x)
                    x = PReLU(shared_axes=[1, 2])(x)
                    #x = _prelu(x,"d"+str(l))
                    print('D-Block: ', x.get_shape())
                    # ttyadd: drop mentioned in paper
                    if l%3 == 2:
                      x = Dropout(rate=0.2)(x)
                    downsampling_l.append(x)

            # bottleneck layer
            with tf.compat.v1.name_scope('bottleneck_conv'):
                x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1],
                        activation=None, padding='same', kernel_initializer=Orthogonal(),
                        strides=2))(x)
                # ttyadd: prelu mentioned in paper
                #x = LeakyReLU(0.2)(x)
                x = PReLU(shared_axes=[1, 2])(x)
                #x = _prelu(x,"b"+str(l))
                x = Dropout(rate=0.2)(x)
                # ttyadd: print
                print('B-Block: ', x.get_shape())

            # upsampling layers
            for l, nf, fs, l_in in reversed(list(zip(list(range(L)), n_filters, n_filtersizes, downsampling_l))):
                with tf.compat.v1.name_scope('upsc_conv%d' % l):
                    # (-1, n/2, 2f)
                    x = (Conv1D(filters=2*nf, kernel_size=fs,
                            activation=None, padding='same', kernel_initializer=Orthogonal()))(x)
                    x = SubPixel1D(x, r=2)
                    # ttyadd: prelu ,mentioned in paper
                    #x = Activation('relu')(x)
                    x = PReLU(shared_axes=[1, 2])(x)
                    #x = _prelu(x,"u"+str(l))
                    # (-1, n, f)
                    # (-1, n, 2f)
                    # ttyadd: drop condition
                    if l % 3 == 2:
                      x = Dropout(rate=0.2)(x)
                    x = K.concatenate(tensors=[x, l_in], axis=2)
                    print('U-Block: ', x.get_shape())

            # final conv layer
            with tf.compat.v1.name_scope('lastconv'):
                x = Conv1D(filters=2, kernel_size=11,
                        activation='linear', padding='same', kernel_initializer=RandomNormal(stddev=1e-3))(x)
                x = SubPixel1D(x, r=2)
                print(x.get_shape())

            #g = merge([x, X], mode='sum')
            g = add([x,X])

        return g
    # ttyadd
    def predict(self, X):
        print("predicting")
        assert len(X) == 1
        x_sp = spline_up(X, self.r)
        x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
        X = x_sp.reshape((1,len(x_sp),1))
        print((X.shape))
        feed_dict = self.load_batch((X,X), train=False)
        return x_sp, self.sess.run(self.predictions, feed_dict=feed_dict)

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
