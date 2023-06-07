# audio-super-resolution-tf2.4.1

This is an UNOFFICIAL implementation of the audio super-resolution model proposed in 
**H.M. Wang and D.L. Wang, "Towards Robust Speech Super-Resolution"**. 

The code is based on https://github.com/kuleshov/audio-super-res
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
Balanced Corpus:
`100% VCTKS`
`10% TIMIT, IEEE`
`2% VCTKM, WSJ, LIBRI, and Mixed`
![image](https://user-images.githubusercontent.com/44235744/193955233-150e62ac-cfbb-43db-939d-c00f221845b9.png)

### Hyperparameter
Dropout rate = 0.2, Optimization = Adam, Learning rate = 0.0003 is halved if the loss has not improved for 3 consecutive epochs on the validation set. Early stop if the validation loss has not improved for 6 successive epochs.

### Results
Here we provide a qualitative example of some datasets [https://tan90xx.github.io/SR-display.github.io/](https://tan90xx.github.io/SR-display.github.io/) 

<div align=center>

T1. EXPERIMENTAL RESULTS FOR CROSS-CORPUS SR USING THE FOUR BASELINES AND PROPOSED MODEL
|                            |        | TIMIT |       |        | WSJ   |       |        | LIBRI |       |        | IEEE  |       |
|:--------------------------:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
| **Model/Training Dataset** | SNR    | LSD   | PESQ  | SNR    | LSD   | PESQ  | SNR    | LSD   | PESQ  | SNR    | LSD   | PESQ  |
| **Spline**                 | 18.27  | 2.07  | 3.51  | 9.59   | 2.27  | 2.83  | 19.73  | 2.22  | 3.43  | 21.12  | 2.14  | 3.94  |
| **DNN-BWE/TIMIT**          | 17.38  | 1.64  | 1.82  | 7.69   | 1.41  | 1.37  | 19.20  | 1.27  | 1.93  | 20.46  | 1.57  | 1.81  |
| **DNN-BWE/WSJ**            | 17.40  | 1.56  | 1.91  | 7.70   | 1.31  | 1.42  | 19.14  | 1.23  | 2.00  | 20.39  | 1.49  | 1.91  |
| **DNN-BWE/LIBRI**          | 17.52  | 1.57  | 1.89  | 7.75   | 1.37  | 1.43  | 19.37  | 1.21  | 2.03  | 20.68  | 1.53  | 1.86  |
| **DNN-BWE/IEEE**           | 11.55  | 2.52  | 1.20  | -1.12  | 2.40  | 1.06  | 15.39  | 1.96  | 1.31  | 15.92  | 2.31  | 1.23  |
| **DNN-Cepstral/TIMIT**     | 17.23  | 1.02  | 2.43  | 11.18  | 0.81  | 1.79  | 18.97  | 0.85  | 2.68  | 19.84  | 1.02  | 2.59  |
| **DNN-Cepstral/WSJ**       | 17.23  | 1.02  | 2.42  | 11.18  | 0.82  | 1.78  | 18.97  | 0.85  | 2.69  | 19.84  | 1.02  | 2.57  |
| **DNN-Cepstral/LIBRI**     | 16.31  | 1.54  | 1.68  | 9.78   | 1.19  | 1.32  | 18.83  | 1.13  | 2.39  | 19.63  | 1.37  | 1.91  |
| **DNN-Cepstral/IEEE**      | 17.22  | 1.06  | 2.34  | 11.11  | 0.84  | 1.72  | 18.96  | 0.87  | 2.64  | 19.83  | 1.03  | 2.56  |
| **AudioUNet/TIMIT**        | 18.41  | 1.69  | 3.13  | 10.19  | 2.11  | 3.16  | 20.49  | 1.60  | 3.17  | 22.16  | 1.73  | 3.55  |
| **AudioUNet/WSJ**          | 18.35  | 1.92  | 3.40  | 9.90   | 2.23  | 2.72  | 20.49  | 1.76  | 3.47  | 22.19  | 1.94  | 3.74  |
| **AudioUNet/LIBRI**        | 18.44  | 2.01  | 3.62  | 10.19  | 2.27  | 2.96  | 20.82  | 2.01  | 3.85  | 22.62  | 2.20  | 4.01  |
| **AudioUNet/IEEE**         | 18.37  | 1.80  | 3.28  | 10.01  | 2.23  | 3.00  | 20.32  | 1.73  | 3.36  | 21.92  | 1.84  | 3.67  |
| **TFNet/TIMIT**            | 17.08  | 1.18  | 2.91  | 10.18  | 1.29  | 2.38  | 14.86  | 1.30  | 2.35  | 21.04  | 1.33  | 2.71  |
| **TFNet/WSJ**              | 15.17  | 1.18  | 2.40  | 9.27   | 1.26  | 2.27  | 15.36  | 1.28  | 2.22  | 20.31  | 1.45  | 2.09  |
| **TFNet/LIBRI**            | 15.82  | 1.17  | 2.33  | 10.97  | 1.20  | 2.22  | 16.84  | 1.26  | 2.76  | 22.97  | 1.24  | 3.04  |
| **TFNet/IEEE**             | 13.48  | 1.39  | 1.82  | 9.39   | 1.36  | 1.80  | 14.89  | 1.36  | 2.08  | 21.37  | 1.33  | 2.46  |
| **Proposed/TIMIT**         | 18.58  | 1.64  | 3.82  | 10.68  | 1.91  | 3.58  | 21.33  | 1.58  | 3.76  | 23.27  | 1.70  | 4.22  |
| **Proposed/WSJ**           | 18.48  | 1.50  | 3.02  | 10.39  | 1.68  | 2.44  | 21.00  | 1.41  | 3.36  | 17.44  | 1.53  | 2.69  |
| **Proposed/LIBRI**         | 18.55  | 1.87  | 3.41  | 10.57  | 2.09  | 2.59  | 21.31  | 1.83  | 3.79  | 23.22  | 1.98  | 3.82  |
| **Proposed/IEEE**          | 18.29  | 1.50  | 2.54  | -5.97  | 1.70  | 1.11  | 17.11  | 1.32  | 2.27  | 1.47   | 1.46  | 1.27  |
| **Proposed/Mixed**         | 18.48  | 1.74  | 3.21  | 5.64   | 1.94  | 1.41  | 21.29  | 1.69  | 3.63  | 23.09  | 1.84  | 3.68  |


T2. COMPARISON OF VARIOUS LOSS FUNCTIONS ON THE TIMIT DATASET
| **LOSS**   | SNR    | LSD   | PESQ  |
|:----------:|:------:|:-----:|:-----:|
| **MAE**    | 18.48  | 1.70  | 3.33  |
| **MSE**    | 18.48  | 1.49  | 2.78  |
| **F**      | 18.50  | 1.61  | 3.11  |
| **RI**     | 18.58  | 1.73  | 3.53  |
| **TF**     | 18.44  | 1.69  | 3.00  |
| **RI-MAG** | 18.49  | 1.73  | 3.20  |
| **PCM**    | 18.41  | 1.48  | 2.77  |
| **T-PCM**  | 18.58  | 1.64  | 3.82  |


T3. MODEL TRAINED ON ORIGINAL TIMIT UTTERANCES TESTED ON DATA CONVOLVED WITH DIFFERENT MIRs
| **Modle**              | SNR     | LSD     | PESQ     |
|:----------------------:|:-------:|:-------:|:--------:|
| **Spline**             | 18.27   | 2.07    | 3.51     |
| **Original**           | 18.58   | 1.64    | 3.82     |
| **Test on MIR1**       | 11.70   | 1.66    | 3.50     |
| **Test on MIR2**       | 14.53   | 1.51    | 3.71     |
| **Average of 20 MIRs** | 15.33   | 1.58    | 3.70     |


T4. EXPERIMENTAL RESULTS FOR SR MODELS EVALUATED ON VCTK WITH DOWNSAMPLING FACTOR FOR 2 AND 4
|                  |       |        | **VCTKS** |   |  **VCTKM**  |  |       |
|:----------------:|:-----:|:------:|:-----:|:-----:|:------:|:-----:|:-----:|
| **Model**        | R     | SNR    | LSD   | PESQ  | SNR    | LSD   | PESQ  |
| **Spline**       | 2.00  | 19.42  | 2.13  | 3.06  | 22.40  | 1.96  | 3.92  |
| **DNN-BWE**      | 2.00  | 19.12  | 1.49  | 2.07  | 17.48  | 2.23  | 1.79  |
| **DNN-Cepstral** | 2.00  | 18.59  | 0.89  | 3.08  | 19.62  | 1.45  | 2.69  |
| **AudioUNet**    | 2.00  | 20.21  | 1.52  | 2.79  | 22.54  | 1.77  | 3.85  |
| **TFNet**        | 2.00  | 22.00  | 1.50  | 2.41  | -2.22  | 2.59  | 1.05  |
| **Proposed**     | 2.00  | 14.98  | 1.44  | 2.77  | 23.01  | 1.73  | 4.06  |
| **Spline**       | 4.00  | 15.28  | 3.01  | 3.16  | 19.28  | 2.64  | 3.36  |
| **DNN-BWE**      | 4.00  | 15.01  | 1.72  | 1.72  | 18.61  | 2.22  | 1.66  |
| **DNN-Cepstral** | 4.00  | 15.16  | 1.36  | 2.62  | 18.28  | 1.66  | 2.30  |
| **AudioUNet**    | 4.00  | 15.50  | 2.08  | 2.30  | 19.47  | 2.26  | 2.56  |
| **TFNet**        | 4.00  | 13.41  | 1.74  | 1.87  | 15.20  | 1.61  | 2.15  |
| **Proposed**     | 4.00  | 15.43  | 2.12  | 2.62  | 20.30  | 2.12  | 3.41  |

</div>


### Changelog
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
<div align=center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189524252-b1a8a0ce-616f-46ec-b8f5-877dfb99df1a.png" width="400">
    
    Fig from P2060 of the paper
</div>

- (7)Self-define loss function. 
(OLA)"Framed segments are first divided into frames of 512 samples with a frame shift of 256 samples. Then we multiply these frames with a Hamming window."

``` 
# ola output[batch, length]
x_ola = tf.signal.overlap_and_add(X, 1024)
x_ola = tf.cast(x_ola, tf.float32)
X_spec = tf.signal.stft(signals=x_ola, frame_length=FRAME, frame_step=SHIFT, fft_length=FRAME,
                        window_fn=tf.signal.hamming_window)
``` 

<!-- - (8)Apply OLA(overlap and add). 
Also pay attention to the articulation of each two wav files, which is not continuous. Some frames between them have been discarded to avoid vertical stripes in spectrogram. This easy numpy version can be put to calculate LSD and PESQ. -->

<div align=center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385562-6f7a2bcc-5940-45be-8157-9a2ae282088f.png">
    
    Fig1.Flat
</div>
<div align=center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385109-af53b752-b293-4a30-b8ce-6a9e9a2e37ab.png">
    
    Fig2.OLA
</div>
<!--Fig3.Discard
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="https://user-images.githubusercontent.com/44235744/189385130-7ba1448b-e704-4c61-a24d-3753e7736bec.png">
</center> -->

- (8)Apply dequeue to monitor val_loss and achieve early stopping ect.
- (9)Write the logdir and csv records in the same path for quick check.

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
- (10)Design the output path of eval files same with logdir and csv. 
- (11)Auto-calculate metrics.
- (12)Define visualization functions to display Spectrogram and Training process
- (13)Build a web page [[code]](https://github.com/tan90xx/tan90xx.github.io/tree/main/SR-display.github.io).

Slight changes in script , which works as follows.
``` 
optional arguments:
-h, --help              
--logname 
--out-label 
--wav-file-list 
--r R                 
--sr SR          
--ola false

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
### ACKNOWLEDGMENT
I would like to thank the author Heming Wang who have been kind enough to answer my questions about the dimensions of DFT and the silence filter.

[zayd/deep-audio-super-resolution](https://github.com/zayd/deep-audio-super-resolution) - for DNN baseline<br>
[https://github.com/moodoki/tfnet](https://github.com/moodoki/tfnet) - for TFNet baseline

Individual items may not be added in time. If you find out, please let me know. Hope to understand
