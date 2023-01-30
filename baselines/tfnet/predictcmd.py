import os
#os.system("python predict.py --model_dir ./logdir/tfnet2018/ieee/ds2 --wav_file_list D:/Audio-Kuleshov/data/Corpus/timit-test-files.txt")
#os.system("python predict.py --model_dir ./logdir/tfnet2018/libri/ds2")
#os.system("python predict.py --model_dir ./logdir/tfnet2018/timit/ds2")
#os.system("python predict.py --model_dir ./logdir/tfnet2018/wsj/ds2")
os.system("python predict.py --model_dir ./logdir/tfnet2018/vctk_m/ds4 --wav_file_list D:/Audio-Kuleshov/data/Corpus/vctk-multispeaker-test-files.txt")
os.system("python predict.py --model_dir ./logdir/tfnet2018/vctk-p225/ds4 --wav_file_list D:/Audio-Kuleshov/data/Corpus/vctk-multispeaker-test-files.txt")