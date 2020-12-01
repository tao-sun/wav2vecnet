import os, sys

import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_spectrogram(wav_dir, wav_name, save_dir, sample_rate=16000):
    y, sr = librosa.load(os.path.join(wav_dir, wav_name), sr=sample_rate)
    spec = np.abs(librosa.stft(y, 512, 256, 512))
    norm_spec = librosa.power_to_db(spec**2)
    black_time_frames = np.array([28234, 61002, 93770, 126538]) / 256.0

    fig, ax = plt.subplots()
    img = ax.imshow(norm_spec)
    plt.vlines(black_time_frames, [0, 0, 0, 0], [10, 10, 10, 10], colors="red", lw=2, alpha=0.5)
    plt.vlines(black_time_frames, [256, 256, 256, 256], [246, 246, 246, 246], colors="red", lw=2, alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(img, cax=cax)

    ax.xaxis.set_label_position("bottom")
    #ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 256.0 / sr))
    #ax.xaxis.set_major_formatter(ticks_x)
    ax.xaxis.set_major_locator(ticker.FixedLocator(([i * sr / 256. for i in range(len(y)//sr + 1)])))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(([str(i) for i in range(len(y)//sr + 1)])))

    ax.yaxis.set_major_locator(ticker.FixedLocator(([float(i) * 2000.0 / (sr/2.0) * 256. for i in range(6)])))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([str(i*2) for i in range(6)]))

    ax.set_xlabel("t (s)")
    ax.set_ylabel('f (KHz)')

    fig.set_size_inches(7., 3.)
    fig.savefig(os.path.join(save_dir, wav_name + ".png"), bbox_inches='tight')


if __name__ == '__main__':
    audio_dir = sys.argv[1]
    audio_name = sys.argv[2]
    sr = sys.argv[3] if len(sys.argv) > 3 else 16000
    spec_dir = sys.argv[4] if len(sys.argv) > 4 else audio_dir

    draw_spectrogram(audio_dir, audio_name, spec_dir, sr)


