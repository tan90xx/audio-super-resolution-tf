"""Tests for predict mode"""
import sys
import os
import tensorflow.compat.v1 as tf
import soundfile as sf
import numpy as np
import argparse
from tfnet import TFNetEstimator
from tfnet import nets
import datahelper.dataset as ds
from lrdata import get_single_file_dataset

def make_prediction(args):
    """Test running eval from with trained model"""
    tf.logging.set_verbosity(tf.logging.INFO)

    #RunConfig for more more printing since we are only training for very few steps
    config = tf.estimator.RunConfig(log_step_count_steps=1)

    tfnet_est = TFNetEstimator(**nets.default_net(), config=config, model_dir=args.model_dir)
    
    if args.wav_file_list:
        with open(args.wav_file_list) as f:
            for line in f:
                upsample_wav(args.corpus_dir+line.strip(), args,tfnet_est)

def check_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def upsample_wav(LQ_AUDIO_FILE, args, tfnet_est):
    #2 files, 64 epochs, batchsize 32 => 2*64/32 = 4 iterations
    preds = tfnet_est.predict(input_fn=lambda: ds.single_file_dataset(LQ_AUDIO_FILE,).make_one_shot_iterator().get_next())
    result = np.array(list(preds))
    result = result.flatten()

    outname = args.model_dir+'/eval'+LQ_AUDIO_FILE[LQ_AUDIO_FILE.find('Corpus')+6:]
    check_dir(os.path.abspath(os.path.dirname(outname)))
    sf.write(outname+ '.pr.wav',result,int(16000))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Predictions')
    parser.add_argument('--model_dir', type=str, default='./logdir/tfnet2018/vctk-p225/ds2',
                        help='model file to make predictions')
    parser.add_argument('--corpus_dir', type=str, default='D:/Audio-Kuleshov/data/Corpus',
                        help='corpus for predictions')
    parser.add_argument('--wav_file_list', type=str, default='D:/Audio-Kuleshov/data/Corpus/test.txt',
                        help='list of wavs for predictions in model dir')
    args, _ = parser.parse_known_args()

    make_prediction(args)        
