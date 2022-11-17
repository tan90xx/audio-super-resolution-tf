"""dataset for prediction"""
import numpy as np
import tensorflow.compat.v1 as tf
from scipy.signal import decimate, resample
try:
    import librosa
except ImportError:
    tf.logging.warn("librosa is not available, loading form wav files might not work")

def get_single_file_dataset(filename='./tests/audioclips/file1_d2.wav', upsample_rate=2, seg_length=8192, batchsize=16, **kwars):
    def _upsample(x, rate):
        x_us = resample(x, len(x)*rate, axis=0)
        return x_us.astype(x.dtype)
    
    def _audio_to_float(data):
        if data.dtype == np.float32:
            return data
        return np.true_divide(data, np.iinfo(data.dtype).max, dtype=np.float32)

    def _load_wav(filename, trim_silence=None, gt_rate=16000,):
        data, _ = librosa.load(filename, sr=gt_rate)
        if trim_silence:
            data, _ - librosa.effects.trim(data, top_db=trim_silence)
        data = _audio_to_float(data)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        return data


    audio_in = _upsample(_load_wav(filename), upsample_rate)
    audio_len, channels = audio_in.shape
    padlen = seg_length - audio_len%seg_length
    audio_padded = np.pad(audio_in, [(0,padlen), (0, 0)], 'constant')
    audio_segs = audio_padded.reshape((-1, seg_length, channels))
    
    def _gen():
        for seg in audio_segs:
            yield (seg,seg)
    
    dset = tf.data.Dataset.from_generator(_gen,
            output_types = (tf.float32, tf.float32),
            output_shapes = ([seg_length,channels],[seg_length,channels]))
    dset = dset.batch(batchsize)
    return dset

def get_lr_dataset(length=8192, channels=1, count=16,
                      batchsize=16, repeat=200,
                      drop_remainder=True
                     ):
    """lr dataset generator for use in unit tests"""
    lr_hr = np.array(np.linspace(0, 1, length)[:, np.newaxis], dtype=np.float32)
    lr_hr = np.hstack([lr_hr for _ in range(channels)])
    lr_lr = lr_hr.copy()
    lr_lr[1::2] = 0

    lr_train = [(lr_lr.copy(), lr_hr.copy()) for _ in range(count)]

    lr_dset = tf.data.Dataset.from_generator(lambda: ((l, h) for l, h in lr_train),
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=([length, channels],
                                                               [length, channels]))

    #16 samples per epoch, 2 epochs, batch size 4 -> 8 iterations
    lr_dset = lr_dset.repeat(repeat).batch(batchsize)
    return lr_dset
