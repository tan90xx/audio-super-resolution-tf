# audio-super-resolution-tf2.4.1
This is the implementation of the audio super-resolution model proposed in Towards Robust Speech Super-Resolution with no baselines, only the proposed model.
### Environment
!pip install https://github.com/schmiph2/pysepm/archive/master.zip

### Network
`Parameters: 10281363`
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
`Kernel size = 11`

### Dataset
10% Corpus: TIMIT, VCTKS, VCTKM, WSJ, LIBRI, IEEE, and Mixed.

### Hyperparameter
Dropout rate = 0.2, Optimization = Adam, Learning rate = 0.0003 is halved if the loss has not improved for 3 consecutive epochs on the validation set. Early stop if the validation loss has not improved for 6 successive epochs.

### Result
