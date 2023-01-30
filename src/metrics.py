import librosa
import numpy as np
np.seterr(divide = 'ignore')
import pysepm
import plotting
import os
################################################ loop ##################################################################
def main(corpus_path = r"D:\Audio-Kuleshov\data\Corpus",
         check_dir_path = r"D:\Audio-Kuleshov\src\checkpoints",
         corpus=['IEEE','LIBRI','TIMIT','WSJ'],
         down_scheme = ['d','s','f','r'],
         spline = False,
         a = None,
         ):
    '''
    Strings below are for the right dir path.
    Args:
        corpus_path:    to get the ground true wav
        check_dir_path: to get the predicted wav
        corpus:         name of corpus tested
        down_scheme:    name of the downsample scheme before predictation
        spline:         to offer 'sp' for high path location
        a:              name list of test dirs
    Returns:
        None
    '''
    # setting path
    if a == None:
        os.chdir(check_dir_path)
        a = [x for x in os.listdir() if os.path.isdir(x)]
    #######################################################
    def find_wavs(input_path):
        # find all wav files
        flac_files = []
        for root, folders, files in os.walk(input_path):
            for file in filter(lambda x: x.endswith('wav'), files):
                if '._S_' in file:# for IEEE
                    continue
                else:
                    flac_files.append(os.path.join(root, file))
        return flac_files

    def find_high_path(super_path):
        p1 = super_path.find('eval')
        if spline==True:
            p2 = super_path.find('.wav.sp')
        else:
            p2 = super_path.find('.wav.p')
        high_path = os.path.join(corpus_path,super_path[p1+7:p2]+'.wav')
        return high_path
    ########################################################
    for dir_name in a:
        for corp_name in corpus:
            for down in down_scheme:
                path = check_dir_path+"\{}\eval-{}\{}".format(dir_name,down,corp_name)
                wavlist = find_wavs(path)
                SNR1, SNR2, LSD, PESQ=list(), list(), list(), list()
                for super_path in wavlist:
                    ####### generate_path ##########
                    high_path = find_high_path(super_path)
                    #print(high_path)
                    ######### load audio ##########
                    High, _ = librosa.load(high_path, sr=16000)
                    Super, _ = librosa.load(super_path, sr=16000)
                    MIN = min(len(High),len(Super))
                    High = High[:MIN]
                    Super = Super[:MIN]
                    ######### cumpute ############
                    temp_snr1, temp_snr2, temp_lsd, temp_pesq = get_scores(High, Super)
                    SNR1.append(temp_snr1)
                    SNR2.append(temp_snr2)
                    LSD.append(temp_lsd)
                    PESQ.append(temp_pesq)
                print("{:<20}{:<15}{:<25}{:<25}{:<25}{:<25}".format(dir_name,'{}-{}'.format(corp_name,down),np.mean(SNR2),np.mean(SNR1),np.mean(LSD),np.mean(PESQ)))
################################################ cal  func #############################################################
def compute_snr1(x, x_hat):

    eps = np.finfo(np.float64).eps
    x_energy = np.square(np.linalg.norm(x))
    noise_energy = np.square(np.linalg.norm(x_hat-x))
    snr = 10. * np.log10((x_energy / (noise_energy+eps))+eps)

    return snr

def compute_snr2(original_waveform, target_waveform):
    snr = pysepm.SNRseg(original_waveform,target_waveform,16000,0.128,0.5)
    return snr

# stolen from https://github.com/UCLA-ECE209AS-2018W/Weikun-Zhengshuang
def compute_lsd(origianl_waveform, target_waveform):
    """ Compare lsd between the original and target audio

    The log-spectral distance (LSD), also referred to as log-spectral distortion,
    is a distance measure (expressed in dB) between two spectra

    Args:
        param1 (list): origianl_spectrogram
        param2 (list): target_spectrogram

    Returns:
        float: compute lsd for spectrogram plots

    """
    eps = np.finfo(np.float64).eps
    # Compute FFT
    original_spectrogram = np.abs(
        librosa.stft(origianl_waveform, n_fft=512, hop_length=256, win_length=512, window='hann'))
    target_spectrogram = np.abs(
        librosa.stft(target_waveform, n_fft=512, hop_length=256, win_length=512, window='hann', ))
    # Compute power spectra
    original_log = np.log10(np.power(np.abs(original_spectrogram), 2) + eps)
    target_log = np.log10(np.power(np.abs(target_spectrogram), 2) + eps)
    original_target_squared = np.power((original_log - target_log), 2)

    # original_spec = np.power(np.abs(original_spectrogram),2)
    # target_spec = np.power(np.abs(target_spectrogram),2)+eps
    # original_target_squared = np.power(np.log10(original_spec / target_spec),2)

    target_lsd = np.mean(np.sqrt(np.mean(original_target_squared, axis=0)))

    return target_lsd


def compute_pesq(x, x_hat, fs=16000):
    return pysepm.pesq(x, x_hat, fs=fs)[-1]

def get_scores(High, Super):
    return compute_snr1(High, Super),compute_snr2(High, Super),compute_lsd(High, Super),compute_pesq(High, Super)

#########################################################################################
def check(wavlist,i):
    super_path = wavlist[i]
    high_path = find_high_path(wavlist[i])
    High, fs = librosa.load(high_path, sr=16000)
    Super, fs = librosa.load(super_path, sr=16000)
    plotting.draw_spec(High, Super,a=12,b=6)
