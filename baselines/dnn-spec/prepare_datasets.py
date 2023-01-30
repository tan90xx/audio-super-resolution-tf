from experiments import experiments
#from sklearn.externals import joblib
import joblib

from feature_extraction import FeatureExtraction
import os


SPEAKER1_TRAIN = 'D:/Audio-Kuleshov/data/Corpus/vctk-speaker1-train-files.txt'
SPEAKER1_VAL = 'D:/Audio-Kuleshov/data/Corpus/vctk-speaker1-val-files.txt'
SPEAKER1_DATA = 'D:/Audio-Kuleshov/data/Corpus/'

MULTISPEAKER_TRAIN = 'D:/tfnet/balanced_corpus/vctk-multispeaker-train-files.txt'
MULTISPEAKER_VAL = 'D:/tfnet/balanced_corpus/vctk-multispeaker-val-files.txt'
MULTISPEAKER_DATA = 'D:/Audio-Kuleshov/data/Corpus/'

MUSIC_TRAIN = '../../data/music/music_train.npy'
MUSIC_VAL = '../../data/music/music_valid.npy'
MUSIC_DATA = ''

OUTPUT_DIR = './output/'

def create_path(params):
  path = ''
  for key in params.keys():
    path += key + '=' + str(params[key]) + '/'
  return path

# Loop over the experiments create necessary datasets and save to paths
for experiment in experiments:
    if experiment['dataset'] == 'speaker1':
      fe = FeatureExtraction(train_files=SPEAKER1_TRAIN,
                             val_files=SPEAKER1_VAL,
                             data_dir=SPEAKER1_DATA,
                             dataset='vctk',
                             upsample=experiment['upsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    elif experiment['dataset'] == 'multispeaker':
      fe = FeatureExtraction(train_files=MULTISPEAKER_TRAIN,
                             val_files=MULTISPEAKER_VAL,
                             data_dir=MULTISPEAKER_DATA,
                             dataset='vctk',
                             upsample=experiment['upsample'],
                             train_subsample=experiment['subsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    elif experiment['dataset'] == 'music':
      fe = FeatureExtraction(train_files=MUSIC_TRAIN,
                             val_files=MUSIC_VAL,
                             data_dir=MUSIC_DATA,
                             dataset='music',
                             upsample=experiment['upsample'])

      SAVE_DIR = OUTPUT_DIR + create_path(experiment)

    print("Saving output to:", SAVE_DIR)
    os.makedirs(SAVE_DIR)
    joblib.dump(fe, SAVE_DIR + 'fe')
