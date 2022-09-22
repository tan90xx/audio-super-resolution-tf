import os
import time

import numpy as np
import tensorflow as tf
import pickle

#import librosa
from tensorflow.python.keras import backend as K
from .dataset import DataSet
from tqdm import tqdm

# ttyadd: lib
#from pysepm import pesq
#import soundfile as sf
from collections import deque
import csv
from tensorflow.python.ops import array_ops
import functools

#window_fn = functools.partial(tf.signal.hamming_window, periodic=True)
# ----------------------------------------------------------------------------
# Warning:
# initialize_all_variables-->global_variables_initializer
# ----------------------------------------------------------------------------
# ttynote: mostly not use and we change opt in run.get_model
default_opt = {'loss_func': 'L2', 'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
               'layers': 2, 'batch_size': 128}


class Model(object):
    """Generic tensorflow model training code"""

    def __init__(self, from_ckpt=False, n_dim=None, r=2,
                 opt_params=default_opt, log_prefix='./run'):

        # create session
        self.sess = tf.compat.v1.Session()
        K.set_session(self.sess)  # pass keras the session

        # save params
        self.opt_params = opt_params
        self.layers = opt_params['layers']

        # ttyadd: loss monitor
        self.log_prefix = log_prefix
        if 'proposed' in self.log_prefix:
            self.loss_deque = deque(maxlen=6)

        if from_ckpt:
            pass  # we will instead load the graph from a checkpoint
        else:
            # create input vars
            X = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1), name='X')#ttyadd: reference
            Y = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1), name='Y')#        prediction
            alpha = tf.compat.v1.placeholder(tf.float32, shape=(),
                                   name='alpha')  # weight multiplier
            # save inputs
            self.inputs = (X, Y, alpha)
            tf.compat.v1.add_to_collection('inputs', X)
            tf.compat.v1.add_to_collection('inputs', Y)
            tf.compat.v1.add_to_collection('inputs', alpha)

            # create model outputs
            self.predictions = self.create_model(n_dim, r)
            tf.compat.v1.add_to_collection('preds', self.predictions)
            # init the model
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)

            # create training updates
            self.train_op = self.create_train_op(X, Y, alpha)
            tf.compat.v1.add_to_collection('train_op', self.train_op)
        # logging
        lr_str = '.' + 'lr%f' % opt_params['lr']
        # ttyadd: loss_str
        loss_str = '.' + '{}'.format(opt_params['loss_func'])
        #g_str = '.g%d' % self.layers
        #b_str = '.b%d' % int(opt_params['batch_size'])
        self.logdir = log_prefix + lr_str + loss_str#'.%d' % r + g_str + b_str
        self.checkpoint_root = os.path.join(self.logdir, 'model.ckpt')
        self.csv_path = os.path.join(self.logdir, 'history.csv')

    def get_power(self, x):
        S = librosa.stft(x, 2048)
        p = np.angle(S)
        S = np.log(np.abs(S)**2 + 1e-8)
        return S

    def compute_log_distortion(self, x_hr, x_pr):
        S1 = self.get_power(x_hr)
        S2 = self.get_power(x_pr)
        lsd = np.mean(np.sqrt(np.mean((S1-S2)**2 + 1e-8, axis=1)), axis=0)
        return min(lsd, 10.)

    def create_train_op(self, X, Y, alpha):
        # load params
        opt_params = self.opt_params
        print('creating train_op with params:', opt_params)

        # create loss
        self.loss = self.create_objective(X, Y, opt_params)

        # create params
        params = self.get_params()

        # create optimizer
        self.optimizer = self.create_optimzier(opt_params)

        # create gradients
        grads = self.create_gradients(self.loss, params)

        # create training op
        with tf.compat.v1.name_scope('optimizer'):
            train_op = self.create_updates(params, grads, alpha, opt_params)

        # initialize the optimizer variabLes
        optimizer_vars = [v for v in tf.compat.v1.global_variables() if 'optimizer/' in v.name
                          or 'Adam' in v.name]
        #init = tf.variables_initializer(optimizer_vars)
        #init = tf.compat.v1.initialize_all_variables()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        return train_op

    def create_model(self, n_dim, r):
        raise NotImplementedError()


    def create_objective(self, X, Y, opt_params):
        # load model output and true output
        P = self.predictions
        ############################################################################################
        # ola output[batch, length]
        x_ola = tf.signal.overlap_and_add(X, 1024)
        x_ola = tf.cast(x_ola, tf.float32)
        y_ola = tf.signal.overlap_and_add(Y, 1024)
        y_ola = tf.cast(y_ola, tf.float32)
        p_ola = tf.signal.overlap_and_add(P, 1024)
        p_ola = tf.cast(p_ola, tf.float32)
        ###########################################################################################
        X_spec = tf.signal.stft(signals=x_ola, frame_length=512, frame_step=256, fft_length=512, window_fn=tf.signal.hamming_window)
        Y_spec = tf.signal.stft(signals=y_ola, frame_length=512, frame_step=256, fft_length=512, window_fn=tf.signal.hamming_window)
        P_spec = tf.signal.stft(signals=p_ola, frame_length=512, frame_step=256, fft_length=512, window_fn=tf.signal.hamming_window)
        ############################################################################################
        '''
        # compute l2 loss
        sqrt_l2_loss = tf.sqrt(tf.reduce_mean(input_tensor=(P-Y)**2 + 1e-6, axis=[1, 2]))
        avg_sqrt_l2_loss = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
        '''
        if opt_params['loss_func'] == 'MAE': 
            sqrt_l2_loss = tf.sqrt(tf.reduce_mean(input_tensor=(P-Y)**2 + 1e-6, axis=[1, 2]))
            avg_sqrt_l2_loss = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
            LOSS = avg_sqrt_l2_loss
        elif opt_params['loss_func'] == 'MSE':
            sqrt_l2_loss = tf.reduce_mean(input_tensor=(P-Y)**2 + 1e-6, axis=[1, 2])
            avg_sqrt_l2_loss = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
            LOSS = avg_sqrt_l2_loss
        elif opt_params['loss_func'] == 'F':
            LOSS = tf.reduce_mean(tf.abs(tf.abs(P_spec) -tf.abs(Y_spec)))
        elif opt_params['loss_func'] == 'RI':
            Y_real_spec = tf.math.real(Y_spec)
            P_real_spec = tf.math.real(P_spec)
            Y_imag_spec = tf.math.imag(Y_spec)
            P_imag_spec = tf.math.imag(P_spec)
            LOSS = tf.reduce_mean(tf.abs(P_real_spec-Y_real_spec) + tf.abs(P_imag_spec-Y_imag_spec))
        elif opt_params['loss_func'] == 'TF':
            #avg_LT = tf.abs(tf.reduce_mean(p_ola-y_ola))
            sqrt_l2_loss = tf.sqrt(tf.reduce_mean(input_tensor=(P-Y)**2, axis=[1, 2]))
            avg_LT = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
            avg_LF = tf.reduce_mean(tf.abs(tf.abs(P_spec) -tf.abs(Y_spec)))
            LOSS = 0.85*avg_LT + 0.15*avg_LF
        elif opt_params['loss_func'] == 'RI_MAG':
            avg_LF = tf.abs(tf.reduce_mean(tf.abs(P_spec) -tf.abs(Y_spec)))
            Y_real_spec = tf.math.real(Y_spec)
            P_real_spec = tf.math.real(P_spec)
            Y_imag_spec = tf.math.imag(Y_spec)
            P_imag_spec = tf.math.imag(P_spec)
            avg_RI = tf.reduce_mean(tf.abs(P_real_spec-Y_real_spec) + tf.abs(P_imag_spec-Y_imag_spec))
            LOSS = avg_LF + avg_RI
        elif opt_params['loss_func'] == 'PCM':
            LOSS = tf.reduce_mean(tf.abs(tf.abs(P_spec) -tf.abs(Y_spec))) + tf.reduce_mean(tf.abs(tf.abs(P_spec - X_spec) - tf.abs(Y_spec - X_spec)))
        else:
            #sqrt_l2_loss = tf.sqrt(tf.reduce_mean(input_tensor=(P-Y)**2 + 1e-6, axis=[1, 2]))
            #avg_LT = tf.reduce_mean(input_tensor=sqrt_l2_loss, axis=0)
            #avg_LT = tf.reduce_mean(tf.abs(p_ola-y_ola))
            #avg_LPCM =  tf.reduce_mean(tf.abs(tf.abs(P_spec) -tf.abs(Y_spec)) +tf.abs(tf.abs(P_spec - X_spec) - tf.abs(Y_spec - X_spec)))
            LOSS = 0.6 * tf.reduce_mean(tf.abs(p_ola-y_ola)) + 0.4 * (tf.reduce_mean(tf.abs(tf.abs(P_spec) -tf.abs(Y_spec)) + tf.abs(tf.abs(P_spec - X_spec) - tf.abs(Y_spec - X_spec)))) 

        # track losses
        tf.compat.v1.summary.scalar('loss', LOSS)

        # save losses into collection
        tf.compat.v1.add_to_collection('losses', LOSS)

        # save predicted and real outputs to collection
        y_flat = tf.reshape(Y, [-1])
        p_flat = tf.reshape(P, [-1])
        tf.compat.v1.add_to_collection('hrs', y_flat)
        tf.compat.v1.add_to_collection('hrs', p_flat)

        return LOSS

    def get_params(self):
        return [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                if 'soundnet' not in v.name]

    def create_optimzier(self, opt_params):
        if opt_params['alg'] == 'adam':
            lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
            optimizer = tf.compat.v1.train.AdamOptimizer(lr, b1, b2)
        else:
            raise ValueError('Invalid optimizer: ' + opt_params['alg'])

        return optimizer

    def create_gradients(self, loss, params):
        gv = self.optimizer.compute_gradients(loss, params)
        g, v = list(zip(*gv))
        return g

    def create_updates(self, params, grads, alpha, opt_params):
        # create a variable to track the global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # update grads
        grads = [alpha*g for g in grads]

        # use the optimizer to apply the gradients that minimize the loss
        gv = list(zip(grads, params))
        train_op = self.optimizer.apply_gradients(
            gv, global_step=self.global_step)

        return train_op

    def load(self, ckpt):
        # get checkpoint name
        if os.path.isdir(ckpt):
            checkpoint = tf.train.latest_checkpoint(ckpt)
        else:
            checkpoint = ckpt
        meta = checkpoint + '.meta'

        # load graph
        self.saver = tf.compat.v1.train.import_meta_graph(meta)
        g = tf.compat.v1.get_default_graph()

        # load weights
        self.saver.restore(self.sess, checkpoint)

        # get graph tensors
        X, Y, alpha = tf.compat.v1.get_collection('inputs')

        # save tensors as instance variables
        self.inputs = X, Y, alpha
        self.predictions = tf.compat.v1.get_collection('preds')[0]

        # load existing loss, or erase it, if creating new one
        g.clear_collection('losses')

        # or, get existing train op:
        self.train_op = tf.compat.v1.get_collection('train_op')
    '''
    def calc_snr(self, Y, Pred):
        sqrt_l2_loss = np.sqrt(np.mean((Pred-Y)**2+1e-6, axis=(0, 1)))
        sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(0, 1)))
        snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
        return snr

    def calc_snr2(self, Y, P):
        sqrt_l2_loss = np.sqrt(np.mean((P-Y)**2 + 1e-6, axis=(1, 2)))
        sqrn_l2_norm = np.sqrt(np.mean(Y**2, axis=(1, 2)))
        snr = 20 * np.log(sqrn_l2_norm / sqrt_l2_loss + 1e-8) / np.log(10.)
        avg_snr = np.mean(snr, axis=0)
        return avg_snr
    '''
    def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=100, r=4, speaker="single", grocery="false", piano="false", calc_full_snr=False):
        # initialize log directory
        if tf.io.gfile.exists(self.logdir):
            tf.io.gfile.rmtree(self.logdir)
        tf.io.gfile.makedirs(self.logdir)

        # load some training params
        n_batch = self.opt_params['batch_size']

        # create saver
        self.saver = tf.compat.v1.train.Saver()

        # summarization
        summary = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(self.logdir, self.sess.graph)

        # load data into DataSet
        train_data = DataSet(X_train, Y_train)
        val_data = DataSet(X_val, Y_val)

        # init np array to store results
        results = np.empty([n_epoch, 6])
        # train the model
        epoch_start_time = time.time()
        total_start_time = time.time()
        step, epoch = 0, train_data.epochs_completed

        print(("Parameters: " + str(count_parameters())))
        # ttyadd: csv
        fieldnames = ['tr_loss', 'va_loss', 'tr_l2_snr', 'va_l2_snr', 'tr_lsd', 'va_lsd', 'tr_pesq', 'va_pesq']
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        while train_data.epochs_completed < n_epoch:

            step += 1

            # load the batch
            alpha = 1.0
            batch = train_data.next_batch(n_batch)
            feed_dict = self.load_batch(batch, alpha)

            # take training step
            tr_objective = self.train(feed_dict)

            # log results at the end of each epoch
            if train_data.epochs_completed > epoch:
                epoch = train_data.epochs_completed
                end_time = time.time()

                tr_l2_loss, tr_l2_snr, tr_lsd, tr_pesq = self.eval_err(
                    X_train, Y_train, n_batch=n_batch)
                va_l2_loss, va_l2_snr, va_lsd, va_pesq = self.eval_err(
                    X_val, Y_val, n_batch=n_batch)

                print("Epoch {} of {} took {:.3f}s ({} minibatches)".format(
                    epoch, n_epoch, end_time - epoch_start_time, len(X_train) // n_batch))
                print("  training loss/segsnr/LSD/PESQ:\t\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    tr_l2_loss, tr_l2_snr, tr_lsd, tr_pesq))
                print("  validation loss/segsnr/LSD/PESQ:\t\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(
                    va_l2_loss, va_l2_snr, va_lsd, va_pesq))

                with open(self.csv_path, 'a+', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    # writer.writeheader()
                    writer.writerow({'tr_loss': tr_l2_loss, 'va_loss': va_l2_loss,
                                     'tr_l2_snr': tr_l2_snr, 'va_l2_snr': va_l2_snr,
                                     'tr_lsd': tr_lsd, 'va_lsd': va_lsd,
                                     'tr_pesq': tr_pesq, 'va_pesq': va_pesq})

                # compute summaries for overall loss
                objectives_summary = tf.compat.v1.Summary()
                objectives_summary.value.add(
                    tag='tr_l2_loss', simple_value=tr_l2_loss)
                objectives_summary.value.add(
                    tag='tr_l2_snr', simple_value=tr_l2_snr)
                objectives_summary.value.add(
                    tag='va_l2_snr', simple_value=va_l2_snr)
                objectives_summary.value.add(tag='tr_lsd', simple_value=tr_lsd)
                objectives_summary.value.add(tag='va_lsd', simple_value=va_lsd)

                # compute summaries for all other metrics
                summary_str = self.sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(objectives_summary, step)

                # write summaries and checkpoints
                summary_writer.flush()
                self.saver.save(
                    self.sess, self.checkpoint_root, global_step=step)

                # ttyadd: early stopping (ugly but the only method I know)
                if 'proposed' in self.log_prefix:
                    self.loss_deque.append(va_l2_loss)
                    if epoch>=3:
                        # 如果连续三个epoch的loss没有提升，学习率减半
                        loss1, loss2, loss3 = self.loss_deque[0], self.loss_deque[1], self.loss_deque[2]
                        if loss1 <= loss2 and loss2 <= loss3:
                            self.opt_params['lr'] = self.opt_params['lr']/2
                            if epoch >= 6:
                                loss4, loss5, loss6 = self.loss_deque[3], self.loss_deque[4], self.loss_deque[5]
                                if loss4 <= loss5 and loss5 <= loss6:
                                    # 如果连续六个epoch的loss没有提升，早停
                                    break
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue

                # calcuate the full snr (currenty on each epoch)
                full_snr = 0
                if(calc_full_snr and train_data.epochs_completed % 1 == 0 and grocery == 'false'):
                    s1 = ""
                    s2 = ""
                    if piano == "true":
                        s1 = "../piano/interp/full-"
                        s2 = "-piano-interp-val." + \
                            str(r) + '.16000.-1.4096.0.1'
                    elif speaker == "single":
                        s1 = "../data/vctk/speaker1/full-"
                        s2 = "-vctk-speaker1-val." + str(r) + '.16000.-1.4096'
                    elif speaker == "multi":
                        s1 = "../data/vctk/multispeaker/full-"
                        s2 = "-vctk-multispeaker-interp-val." + \
                            str(r) + '.16000.-1.8192.0.25'
                    full_clips_X = pickle.load(open(s1 + 'data' + s2, 'rb'))
                    full_clips_Y = pickle.load(open(s1 + 'label' + s2, 'rb'))

                    runs = 0

                    for X, Y in zip(full_clips_X, full_clips_Y):
                        X = np.reshape(X, (1, X.shape[0], 1))
                        Y = np.reshape(Y, (1, Y.shape[0], 1))

                        if self.__class__.__name__ == 'DNN':
                            X = X[:, :8192*(X.shape[1]/8192), :]
                            Y = Y[:, :8192*(Y.shape[1]/8192), :]

                        __, snr, __, __ = self.eval_err(X, Y, 1)
                        full_snr += snr
                    print("Full SNR: " + str(full_snr))

    def train(self, feed_dict):
        _, loss = self.sess.run(
            [self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def load_batch(self, batch, alpha=1, train=True):
        X_in, Y_in, alpha_in = self.inputs

        X, Y = batch
        if Y is not None:
            feed_dict = {X_in: X, Y_in: Y, alpha_in: alpha}
        else:
            feed_dict = {X_in: X, alpha_in: alpha}
        # this is ugly, but only way I found to get this var after model reload
        g = tf.compat.v1.get_default_graph()
        k_tensors = [n for n in g.as_graph_def(
        ).node if 'keras_learning_phase' in n.name]
        if k_tensors:
            k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
            feed_dict[k_learning_phase] = train

        return feed_dict
    # ttyadd : ola for wav monitor
    def overlap_and_add_numpy(self, Y, P):
        # input_shape (frames, 2048)
        Y = np.reshape(Y, (-1, 2048))
        P = np.reshape(P, (-1, 2048))
        n = 0
        for y_frame, p_frame in zip(Y,P):
            if n == 0:
                y = y_frame
                p = p_frame
            else:
                y_half = y_frame[1024:]
                p_half = p_frame[1024:]
                y = np.r_[y, y_half]
                p = np.r_[p, p_half]
            n += 1
        return y, p

    def eval_err(self, X, Y, n_batch=128):
        batch_iterator = iterate_minibatches(X, Y, n_batch, shuffle=False) #ttyadd: True-->False for wav monitor
        loss_op= tf.compat.v1.get_collection('losses')
        y_flat, p_flat = tf.compat.v1.get_collection('hrs')

        loss = 0
        tot_loss = 0
        Ys = np.empty([0, 0])
        Preds = np.empty([0, 0])
        b = 0
        for bn, batch in enumerate(batch_iterator):
            feed_dict = self.load_batch(batch, train=False)
            loss, Y, P = self.sess.run(
                [loss_op, y_flat, p_flat], feed_dict=feed_dict)
            tot_loss += loss[0]
            Ys = np.append(Ys, Y)
            Preds = np.append(Preds, P)
            b = bn
        #l2_pesq = pesq(y, p, 16000)[1]

        return tot_loss / (b+1), 0, 0, 0

    def predict(self, X):
        raise NotImplementedError()

# ----------------------------------------------------------------------------
# helpers


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def count_parameters():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim
        total_parameters += var_params
    return total_parameters
