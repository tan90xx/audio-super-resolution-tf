import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import librosa
import numpy as np

def draw_spec(*param, name=None, save=None, a=6, b=5, show=True, bar=False):
    # ------------------dft parameters---------------------
    FFT=16000
    HOP=160
    WIN=512
    FS=16000
    DT=2
    # ------------------ for color--------------------------
    COLOR="coolwarm"
    # ---------------- loop to draw ------------------------
    n = len(param)
    if n > 1:
        fig, axs = plt.subplots(1, n, sharey=True, figsize=(a, b))
        images = []
        for col, signal in enumerate(param):
            ax = axs[col]

            S_signal = librosa.stft(signal, n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
            Mag_signal = np.abs(S_signal)
            Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

            Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
            X = np.arange(0,np.shape(Mag_signal_db)[1]/FS,1/FS)

            pcm = ax.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

            ax.set_xlim([0,np.shape(Mag_signal_db)[1]/FS])
            images.append(pcm)
            ax.label_outer()
            ax.set_ylim([1, int(FFT / 2 + 1)])
            ax.set_xlabel('Time (s)')
            ax.grid(False)
            if name:
                ax.set_title("{}".format(name[col]))
            if col == 0:
                ax.set_ylabel('Frequency (Hz)')

        # Find the min and max of all colors for use in setting the color scale.
        if bar:
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            for im in images:
                im.set_norm(norm)
            fig.colorbar(images[0], ax=axs, orientation='horizental', fraction=.05, label='Magnitude(dB)')

    else:
        if np.ndim(param[0]) == 2:
            pcm = plt.imshow(param[0], cmap=COLOR)
        else:
            plt.figure(figsize=(a, b))
            S_signal = librosa.stft(param[0], n_fft=FFT, hop_length=HOP, win_length=WIN)  # D x T
            Mag_signal = np.abs(S_signal)
            Mag_signal_db = librosa.amplitude_to_db(Mag_signal)

            Y = np.arange(0,np.shape(Mag_signal_db)[0],1)
            X = np.arange(0,np.shape(Mag_signal_db)[1]/FS,1/FS)

            pcm = plt.pcolormesh(X, Y, Mag_signal_db, shading='auto', cmap="coolwarm")

            plt.xlim([0, np.shape(Mag_signal_db)[1] / FS])

        if name:
            plt.title("{}".format(name[0]))
        plt.ylim([0, int(FFT / 2 + 1)])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(False)
        plt.colorbar(pcm, label='Magnitude(dB)')
    if save:
        plt.savefig('{}.png'.format(save), dpi=600, bbox_inches="tight")
    if show:
        plt.show()



