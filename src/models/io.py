import os
import numpy as np
import h5py
import librosa
import soundfile as sf

from scipy.signal import decimate

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------

def load_h5(h5_path):
    # load training data
    with h5py.File(h5_path, 'r') as hf:
        print('List of arrays in input file:', list(hf.keys()))
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
        print('Shape of X:', X.shape)
        print('Shape of Y:', Y.shape)

    return X, Y
# ttyadd: create dirs if not exist
def check_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

# ttyadd: stolen from https://github.com/deepakbaby/se_relativisticgan
def reconstruct_wav(wavmat, stride_factor=0.5):
    """
    Reconstructs the audiofile from sliced matrix wavmat
    """
    # shape [frames, len]
    window_length = wavmat.shape[1]
    window_stride = int(stride_factor * window_length)
    wav_length = (wavmat.shape[0] - 1) * window_stride + window_length
    wav_recon = np.zeros((1, wav_length))
    # print ("wav recon shape " + str(wav_recon.shape))
    for k in range(wavmat.shape[0]):
        wav_beg = k * window_stride
        wav_end = wav_beg + window_length
        wav_recon[0, wav_beg:wav_end] += wavmat[k, :]

    # now compute the scaling factor for multiple instances
    noverlap = int(np.ceil(1 / stride_factor))
    scale_ = (1 / float(noverlap)) * np.ones((1, wav_length))
    for s in range(noverlap - 1):
        s_beg = s * window_stride
        s_end = s_beg + window_stride
        scale_[0, s_beg:s_end] = 1 / (s + 1)
        scale_[0, -s_beg - 1: -s_end:-1] = 1 / (s + 1)

    return wav_recon * scale_

def upsample_wav(wav, args, model):

    # load signal
    x_hr, fs = librosa.load(wav, sr=args.sr)

    x_lr_t = decimate(x_hr, args.r)

    # pad to mutliple of patch size to ensure model runs over entire sample
    x_hr = np.pad(x_hr, (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 'constant', constant_values=(0,0))

    # downscale signal
    x_lr = decimate(x_hr, args.r)


    # ttyadd: generate patches --> predict --> ola
    if args.ola == 'true':# Mabey not, it seems like worse
        print("++++++++++++++++++++ola++++++++++++++++++++")
        d_lr = int(2048//2)
        s = int(1024//2)
        lr_patches = []
        max_i = len(x_lr) - d_lr + 1
        for i in range(0, max_i, s):
            i_lr = i
            lr_patch = np.array(x_lr[i_lr: i_lr + d_lr])
            lr_patches.append(lr_patch)
        lr_patches = np.array(lr_patches).flatten()
        x_sp, P = model.predict(lr_patches.reshape((1,len(lr_patches),1)))
        x_pr = P.flatten()
        pr_mat = x_pr.reshape((-1,1024))
        x_pr = reconstruct_wav(pr_mat, stride_factor=0.5)
        x_pr = x_pr.flatten()
    # original method
    else:
        x_sp, P = model.predict(x_lr.reshape((1, len(x_lr), 1)))
        # print(np.shape(P))
        x_pr = P.flatten()

    # save the file
    # ttyadd: wav outname
    outname = args.logname+'/eval'+wav[wav.find('Corpus')+6:]+ '.' + args.out_label
    check_dir(os.path.abspath(os.path.dirname(outname)))
    #outname = wav + '.' + args.out_label
    #sf.write(outname + '.lr.wav', x_lr_t, int(fs / args.r))
    #sf.write(outname + '.sp.wav', x_sp, fs) # ttyadd
    #sf.write(outname + '.hr.wav', x_hr, fs)
    sf.write(outname + '.pr.wav', x_pr, fs)

    # save the spectrum
    #S = get_spectrum(x_pr, n_fft=2048)
    #save_spectrum(S, outfile=outname + '.pr.png')
    #S = get_spectrum(x_hr, n_fft=2048)
    #save_spectrum(S, outfile=outname + '.hr.png')
    #S = get_spectrum(x_lr, n_fft=int(2048/args.r))
    #save_spectrum(S, outfile=outname + '.lr.png')

# ----------------------------------------------------------------------------

def get_spectrum(x, n_fft=2048):
    S = librosa.stft(x, n_fft)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    plt.savefig(outfile)
