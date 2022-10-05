"""
Create an HDF5 file of patches for training super-resolution model.
"""

import os, argparse
import numpy as np
import h5py
import pickle

import librosa
from scipy import interpolate
from scipy.signal import decimate, resample

# ttyadd: libs
from tqdm import tqdm
import json
import random
# Set the random seeds
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--file-list',
  help='list of input wav files to process')
parser.add_argument('--in-dir', default='./Corpus/',
  help='folder where input files are located')
parser.add_argument('--out',
  help='path to output h5 archive')
parser.add_argument('--corpus',
  help='folder where input files are located')
parser.add_argument('--state', default='val',
  help='folder where input files are located')
parser.add_argument('--scale', type=int, default=2,
  help='scaling factor')
parser.add_argument('--dimension', type=int, default=2048,
  help='dimension of patches--use -1 for no patching')
parser.add_argument('--stride', type=int, default=1024,
  help='stride when extracting patches')
parser.add_argument('--interpolate', default=True,
  help='interpolate low-res patches with cubic splines')
parser.add_argument('--low-pass', default='subsampling',
  help='apply low-pass filter when generating low-res patches')
parser.add_argument('--batch-size', type=int, default=32,
  help='we produce # of patches that is a multiple of batch size')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')
parser.add_argument('--sam', type=float, default=1,
  help='subsampling factor for the data')

args = parser.parse_args()

# ----------------------------------------------------------------------------

from scipy.signal import butter, cheby1, bessel, lfilter
import re

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# ttyadd: low-pass-filter with three filter types
def Filter(data, cutoff, fs=16000, order=8, filter_type='Cheby1'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if filter_type == 'Butter':
        b, a = butter(order, normal_cutoff)
    elif filter_type == 'Bessel':
        b, a = bessel(order, normal_cutoff)
    else:
        b, a = cheby1(order, 1, normal_cutoff)
    return lfilter(b, a, data)

# ttyadd: subsampling scheme
def subsample(data, rate, fs=16000):
    cutoff = np.floor(fs/(rate*2))
    data = Filter(data, cutoff, fs)
    data = np.array(data[0::rate])
    return data

# ttyadd: silence filter with energy threshold
def silence_filter(data,threshold=0.05):
    # discard the silence
    threshold_db =-10 * np.log10(threshold / 1.0)
    speak, _ = librosa.effects.trim(data, top_db=threshold_db, frame_length=2048, hop_length=1024)
    return speak

# ttyadd: mean and variance normalization
def mvn(data):
    data_zcore = data - np.mean(data)
    data_zcore = data_zcore / np.std(data)
    return data_zcore

def add_data(h5_file, inputfiles, args, save_examples=False):
  # Make a list of all files to be processed
  file_list = []
  # ttyadd: ID is only for vctk-speaker1 corpus
  if 'vctk-speaker1' in args.corpus:
    ID_list = []
  file_extensions = set(['.wav'])
  with open(inputfiles) as f:
    for line in f:
      filename = line.strip()
      ext = os.path.splitext(filename)[1]
      if ext in file_extensions:
        # ttyadd: keeping only a fraction of all the patches still occupies much memory
        u = np.random.uniform()
        if u > args.sam: continue
        file_list.append(os.path.join(args.in_dir, filename))

  num_files = len(file_list)

  # patches to extract and their size
  if args.dimension != -1:
    if args.interpolate:
        d, d_lr = args.dimension, args.dimension
        s, s_lr = args.stride, args.stride
    else:
        d, d_lr = args.dimension, args.dimension / args.scale
        s, s_lr = args.stride, args.stride / args.scale
  hr_patches, lr_patches = list(), list()

  #print(len(file_list))
  # ttyadd: tqdm bar to monitor
  file_list_tqdm = tqdm(file_list, ncols=100)
  for j, file_path in enumerate(file_list_tqdm):
    #if j % 10 == 0: print('%d/%d' % (j, num_files))
    file_list_tqdm.set_description('{0: <40}'.format(args.out))
    if 'vctk-speaker1' in args.corpus:
        directory_id_matches = re.search(fr'p(\d{{3}})\{os.path.sep}', file_path)
        ID = int(directory_id_matches.group(1))

    # load audio file
    x, fs = librosa.load(file_path, sr=args.sr)

    # ttyadd: changes in preprocessing
    x = silence_filter(x)
    x = mvn(x)

    # crop so that it works with scaling ratio
    x_len = len(x)
    x = x[ : x_len - (x_len % args.scale)]

    # ttyadd: changes in preprocessing with three downsample schemes
    schemes = {'subsampling': 1, 'decimating': 2, 'fft': 3, 'random': 4}
    scheme_num = schemes.get(args.low_pass)
    if scheme_num == 4:
        scheme_num = random.randint(1, 3)
    if scheme_num == 1:
        x_lr = subsample(x, args.scale)
    elif scheme_num == 2:
        x_lr = decimate(x, args.scale)
    elif scheme_num == 3:
        x_lr = resample(x, int(np.floor(len(x) / args.scale)))
    # generate low-res version
    #if args.low_pass:
    #  x_lr = decimate(x, args.scale)
    #else:
    #  x_lr = np.array(x[0::args.scale])

    if args.interpolate:
      x_lr = upsample(x_lr, args.scale)
      assert len(x) % args.scale == 0
      assert len(x_lr) == len(x)
    else:
      assert len(x) % args.scale == 0
      assert len(x_lr) == len(x) / args.scale

    if args.dimension != -1:
        # generate patches
        max_i = len(x) - d + 1
        for i in range(0, max_i, s):
            # keep only a fraction of all the patches
            #u = np.random.uniform()
            #if u > args.sam: continue

            if args.interpolate:
                i_lr = i
            else:
                i_lr = i / args.scale

            hr_patch = np.array( x[i : i+d] )
            lr_patch = np.array( x_lr[i_lr : i_lr+d_lr] )

            assert len(hr_patch) == d
            assert len(lr_patch) == d_lr

            hr_patches.append(hr_patch.reshape((d,1)))
            lr_patches.append(lr_patch.reshape((d_lr,1)))
            if 'vctk-speaker1' in args.corpus:
                ID_list.append(ID)
    else: # for full snr
        # append the entire file without patching
        x = x[:,np.newaxis]
        x_lr = x_lr[:,np.newaxis]
        hr_patches.append(x[:len(x) // 256 * 256])
        lr_patches.append(x_lr[:len(x_lr) //256 * 256])
        if 'vctk-speaker1' in args.corpus:
            ID_list.append(ID)

  if args.dimension != -1:
    # crop # of patches so that it's a multiple of mini-batch size
    num_patches = len(hr_patches)
    num_to_keep = int(np.floor(num_patches / args.batch_size) * args.batch_size)
    hr_patches = np.array(hr_patches[:num_to_keep])
    lr_patches = np.array(lr_patches[:num_to_keep])
    if 'vctk-speaker1' in args.corpus:
        ID_list = ID_list[:num_to_keep]
  if save_examples:
    librosa.output.write_wav('example-hr.wav', hr_patches[40][0], fs, norm=False)
    librosa.output.write_wav('example-lr.wav', lr_patches[40][0], fs / args.scale, norm=False)


  if args.dimension != -1:
    # create the hdf5 file
    data_set = h5_file.create_dataset('data', lr_patches.shape, np.float32)
    label_set = h5_file.create_dataset('label', hr_patches.shape, np.float32)

    data_set[...] = lr_patches
    label_set[...] = hr_patches
    if 'vctk-speaker1' in args.corpus:
        #print(len(ID_list))
        pickle.dump(ID_list, open('ID_list_patches_'+str(d)+'_'+str(args.scale), 'wb'))
  else:
    # pickle the data
    pickle.dump(hr_patches, open('full-label-'+args.out[:-7],'wb'))
    pickle.dump(lr_patches, open('full-data-'+args.out[:-7],'wb'))
    if 'vctk-speaker1' in args.corpus:
        pickle.dump(ID_list, open('ID_list','wb'))

# ttyadd: simple rules for uniform naming
def create(args):
    try:
        remark = '-{}'.format(args.low_pass)
        args.out = '{}{}-{}.{}.{}.{}.h5'.format(args.corpus, remark, args.state, args.scale, args.dimension, args.stride)
        args.file_list = './Corpus/{}-{}-files.txt'.format(args.corpus, args.state)

        with h5py.File(args.out, 'w') as f:
            add_data(f, args.file_list, args, save_examples=False)
    except Exception as e:
        print('Error:',repr(e))
        raise


def upsample(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)

  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)

  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

if __name__ == '__main__':
    # ttyadd: batch create
    states = ['train', 'val']
    with open('dataset.json', 'r', encoding='utf8') as fp:
        json_dict = json.load(fp)
    for i in json_dict:
        for state in states:
            args.state = state
            # update from default args
            argparse_dict = vars(args)
            argparse_dict.update(json_dict[str(i)])
            args = argparse.Namespace(**argparse_dict)
            create(args)