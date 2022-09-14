# audio-super-resolution-tf2.4.1
This is an UNOFFICIAL implementation of the audio super-resolution model proposed in 
**H. Wang and D. Wang, "Towards Robust Speech Super-Resolution"** 
with no baselines, only the proposed model. I am in progress of writing this code.

### Environment
!pip install https://github.com/schmiph2/pysepm/archive/master.zip

### Network
`Parameters: 10281363` `Kernel size = 11`
```
D-Block:  (None, None, 64)
D-Block:  (None, None, 64)
D-Block:  (None, None, 64)
D-Block:  (None, None, 128)
D-Block:  (None, None, 128)
D-Block:  (None, None, 128)
D-Block:  (None, None, 256)
D-Block:  (None, None, 256)
B-Block:  (None, None, 256)
U-Block:  (None, None, 512)
U-Block:  (None, None, 512)
U-Block:  (None, None, 256)
U-Block:  (None, None, 256)
U-Block:  (None, None, 256)
U-Block:  (None, None, 128)
U-Block:  (None, None, 128)
U-Block:  (None, None, 128)
```

### Dataset
10% Corpus: TIMIT, VCTKS, VCTKM, WSJ, LIBRI, IEEE, and Mixed.

### Hyperparameter
Dropout rate = 0.2, Optimization = Adam, Learning rate = 0.0003 is halved if the loss has not improved for 3 consecutive epochs on the validation set. Early stop if the validation loss has not improved for 6 successive epochs.

### Result
Notcomplete

### Note
The code is adopted from https://github.com/kuleshov/audio-super-res, and here are some details of changes:
#### Setup
- (1)Set the random seeds and generate txts for train, valid, and test with the path of audio files for each corpus, to repeat this experiments exactly as much as possible.
- (2)Apply MVN and silence filter in datasets preparation, for the reason the author observe that MVN improves cross-corpus generalization, and a silence filter is performed to stabilize training and ensure faster convergence. 
- (3)Extend down-sample schemes in datasets preparation, which used to be only the SciPy decimate function. In summary, the preprocess is like this:
```
-->MVN(mean and variance normalization)
-->Silence filter(discard samples below an energy threshold of 0.05)
  >>>librosa.effects.trim(x,top_db=-20*np.log10(0.05/1.0))
-->Four down-sample schemes to choose: (1)low-pass filter(Chebynov Type I iir filter of order 8)-->Subsampling(discarding samples at fixed intervals) (2)decimate (3)resample (4)mix
-->Upsampling(cubic spline interpolatioin)
-->padding-->generate patches(frames of 2048 samples with overlap of 1024 samples)
```
- (4)Generalize prep_vctk.py to prep_dataset.py for other corpus, and record args in dataset.json for batch generating h5 files.

Slight changes in script , which works as follows.
```
optional arguments: 
-h, --help                  
--corpus
--state                              
--scale                        
--dimension        
--stride                         
--interpolate                 
--low-pass                    
--batch-size           
--sr               
--sam             	

example:
python prep_dataset.py \
  --corpus vctk-speaker1 \
  --state train \
  --scale 2 \
  --dimension 2048 \
  --stride 1024 \
  --interpolate \
  --low-pass decimating\
  --batch-size 32 \
  --sr 16000 \
  --sam 0.25 \
  
delete original arguments:
--file-list                    
--in-dir                        
--out	auto generate by args above:  
--file-list ./Corpus/vctk-speaker1-train-files.txt \
  --in-dir ./Corpus/ \
  --out vctk-speaker1-decimating-train.2.2048.1024.h5 \
```  

#### Running the model
- (5)Make it run. 

``` 
# Wrong:
# Conv1D init-->kernel_initializer, subsample_length-->strides
# Convolution1D-->Conv1D
# stdev-->stddev
# merge-->add
# Warning:
# initialize_all_variables-->global_variables_initializer
``` 

- (6)Build structure of the proposed model, adopted from audiounet. And exactly follow the sequence of Conv-->Relu-->Dropout

``` 
# set:
n_filters = [64, 64, 64, 128, 128, 128, 256, 256]
n_filtersizes = [11, 11, 11, 11, 11, 11, 11, 11, 11]
x = LeakyReLU(0.2)(x)-->x = PReLU(shared_axes=[1])(x)
# add dropout in U and D blocks:
if l%3 == 2:
  x = Dropout(rate=0.2)(x)
``` 
![image](https://user-images.githubusercontent.com/44235744/189524252-b1a8a0ce-616f-46ec-b8f5-877dfb99df1a.png)
Fig from P2060
- (7)Self-define loss function. 
(OLA)"Framed segments are first divided into frames of 512 samples with a frame shift of 256 samples. Then we multiply these frames with a Hamming window."

``` 
# ola output[batch, length]
x_ola = tf.signal.overlap_and_add(X, 1024)
x_ola = tf.cast(x_ola, tf.float32)
X_spec = tf.signal.stft(signals=x_ola, frame_length=FRAME, frame_step=SHIFT, fft_length=FRAME,
                        window_fn=tf.signal.hamming_window)
``` 

- (8)Apply OLA(overlap and add). 
Also pay attention to the articulation of each two wav files, which is not continuous. Some frames between them have been discarded to avoid vertical stripes in spectrogram. This easy numpy version can be put to calculate LSD and PESQ.

Fig1.Flat
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385562-6f7a2bcc-5940-45be-8157-9a2ae282088f.png">
</center>
Fig2.OLA
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385109-af53b752-b293-4a30-b8ce-6a9e9a2e37ab.png">
</center>
Fig3.Discard
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385130-7ba1448b-e704-4c61-a24d-3753e7736bec.png">
</center>

- (9)Apply dequeue to monitor val_loss and achieve early stopping ect.
- (10)Write the logdir and csv records in the same path for quick check.

Slight changes in script , which works as follows.

``` 
optional arguments:
-h, --help   
--model
--loss_func 
--train          
--val             
-e --epochs 
--logname                                                               
--r                                  
--pools_size           
--strides             
--full               	

example:
python run.py train \
  --model proposed \
  --loss_func T_PCM \
  --train ../data/vctk/vctk-speaker1-train.2.2048.1024.h5 \
  --val ../data/vctk/vctk-speaker1-val.2.2048.1024.h5 \
  -e 100 \
  --logname tmp-rum \
  --r 2 \
  --pool_size 2 \
  --strides 2 \
  --full true \
  
delete original arguments:
--alg        
--batch-size  
--layers      
--lr      
--piano   default = false           
--grocery  default = false         
--speaker  default = single	

set in code:
if args.model == 'proposed':
  opt_params = {'loss_func':args.loss_func, 'alg': 'adam', 'lr': 0.0003, 
                'b1': 0.9, 'b2': 0.999, 'batch_size': 32, 'layers': 8}
``` 

#### Testing the model
- (11)Design the output path of eval files same with logdir and csv. 
- (12)Auto-calculate metrics.
- (13)Define visualization functions to display Spectrogram and Training process
- (14)Build a web page [[code]](https://github.com/tan90xx/tan90xx.github.io/tree/main/SR-display.github.io).

Slight changes in script , which works as follows.
``` 
optional arguments:
-h, --help              
--logname 
--out-label 
--wav-file-list 
--r R                 
--sr SR            

example:
python run.py eval \
  --logname ./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-20101 \
  --out-label singlespeaker-out \
  --wav-file-list ../data/vctk/speaker1/speaker1-val-files.txt \
  --r 4 \
  --pool_size 2 \
  --strides 2 \
  --model audiotfilm
``` 
