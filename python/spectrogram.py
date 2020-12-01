#!/usr/bin/env python
# coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import sys, os
from PIL import Image
from tqdm import tqdm

""" short time fourier transform of audio signal """


import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from numpy.lib import stride_tricks

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=16000, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(wavdir, wavname, binsize=2**10, savedir=None, colormap="jet"):
    samplerate, samples = wav.read(os.path.join(wavdir, wavname))

    s = stft(samples, binsize)
    # _, _, s = signal.stft(samples, samplerate, nperseg=binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6, where=np.abs(sshow)>0) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    # plt.colorbar()

    plt.xlabel("Time (s)", fontsize=32)
    plt.ylabel("Frequency (Hz)", fontsize=32)
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    # xlocs = np.float32(np.linspace(0, timebins-1, 5))
    # plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    xlabels = np.arange(0.5, len(samples)/samplerate, step=0.5)
    xlocs = ((xlabels*samplerate - 0.5*binsize) * timebins) / len(samples)
    plt.xticks(xlocs, xlabels, fontsize=24)

    # ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    # plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    ylocs = []
    for i in np.arange(0, 6001, step=2000):
        for j, _ in enumerate(freq):
            if freq[j] == i or (freq[j-1] < i and freq[j] > i):
               ylocs.append(j)
    ylocs.append(len(freq)-1)
    plt.yticks(ylocs, np.arange(0, 8001, step=2000), fontsize=24)

    if savedir:
        savepath = os.path.join(savedir, wavname + ".png")
        plt.savefig(os.path.join(savedir, wavname + ".png"), bbox_inches="tight")
    else:
        savepath = None
        plt.show()

    plt.close()

    return savepath


def plotall(wavdir, savedir, pair=False):
    for _, _, files in os.walk(wavdir):
        for fname in tqdm(files):
            if fname[-4:] == ".wav" and ("clean" not in fname):
                audiopath = plotstft(wavdir, fname, savedir=savedir)
                cleanpath = plotstft(wavdir, fname[:fname.index(".wav")] + "_clean.wav", savedir=savedir)

                if pair:
                    one_image = Image.new('RGB', (1257, 1290))
                    image1 = Image.open(audiopath)
                    image2 = Image.open(cleanpath)

                    one_image.paste(image1, (0, 0))
                    one_image.paste(image2, (0, 645))
                    pairname = fname[:fname.index(".wav")] + "_pair.wav.png"
                    one_image.save(os.path.join(savedir, pairname), 'png')


def plotcombine(wavdir1, wavdir2, savedir1, savedir2, combinedir):
    for _, _, files in os.walk(wavdir1):
        for fname in tqdm(files):
            if fname[-4:] == ".wav" and ("clean" not in fname) and \
                os.path.exists(os.path.join(wavdir1, fname)) and \
                    os.path.exists(os.path.join(wavdir2, fname)):
                audiopath1 = plotstft(wavdir1, fname, savedir=savedir1)
                audiopath2 = plotstft(wavdir2, fname, savedir=savedir2)
                cleanpath = plotstft(wavdir1, fname[:fname.index(".wav")] + "_clean.wav", savedir=savedir1)

                one_image = Image.new('RGB', (1500, 1935))
                image1 = Image.open(audiopath1)
                image2 = Image.open(cleanpath)
                image3 = Image.open(audiopath2)

                one_image.paste(image1, (0, 0))
                one_image.paste(image2, (0, 645))
                one_image.paste(image3, (0, 1290))
                combinename = fname[:fname.index(".wav")] + "_combine.wav.png"
                one_image.save(os.path.join(combinedir, combinename), 'png')


def plot_in_one(model1path, model2path, model3path, model4path, plotpath):
    for db in ['6db', '3db', '0db', 'db-3', 'db-6']:
        for dirpath, dirs, files in os.walk(model1path + '/' + db):
            for dir in dirs:
                os.makedirs(plotpath + '/' + db + '/' + dir)

                plotstft(model1path + '/' + db + '/' + dir + '/input.wav',
                         plotpath=plotpath + '/' + db + '/' + dir + '/input.png')
                plotstft(model1path + '/' + db + '/' + dir + '/gt.wav', plotpath=plotpath + '/' + db + '/' + dir + '/gt.png')

                plotstft(model1path + '/' + db + '/' + dir + '/pred.wav',
                         plotpath=plotpath + '/' + db + '/' + dir + '/pred1.png')
                plotstft(model2path + '/' + db + '/' + dir + '/pred.wav',
                         plotpath=plotpath + '/' + db + '/' + dir + '/pred2.png')
                plotstft(model3path + '/' + db + '/' + dir + '/pred.wav',
                         plotpath=plotpath + '/' + db + '/' + dir + '/pred3.png')
                plotstft(model4path + '/' + db + '/' + dir + '/pred.wav',
                         plotpath=plotpath + '/' + db + '/' + dir + '/pred4.png')

                one_image = Image.new('RGB', (3771, 1290))
                image1 = Image.open(plotpath + '/' + db + '/' + dir + '/input.png')
                image2 = Image.open(plotpath + '/' + db + '/' + dir + '/gt.png')
                image3 = Image.open(plotpath + '/' + db + '/' + dir + '/pred1.png')
                image4 = Image.open(plotpath + '/' + db + '/' + dir + '/pred2.png')
                image5 = Image.open(plotpath + '/' + db + '/' + dir + '/pred3.png')
                image6 = Image.open(plotpath + '/' + db + '/' + dir + '/pred4.png')

                one_image.paste(image1, (0, 0))
                one_image.paste(image2, (1257, 0))
                one_image.paste(image3, (2514, 0))
                one_image.paste(image4, (0, 645))
                one_image.paste(image5, (1257, 645))
                one_image.paste(image6, (2514, 645))
                one_image.save(plotpath + '/' + db + '/' + dir + '/one.png', 'png')


if __name__ == "__main__":
    audio_name = sys.argv[1]

    if audio_name == "all":
        audio_dir = sys.argv[2]
        spec_dir = sys.argv[3] if len(sys.argv) > 3 else audio_dir
        pair = (sys.argv[4] == "pair") if len(sys.argv) > 4 else False
        plotall(audio_dir, savedir=spec_dir, pair=pair)
    elif audio_name == "combine":
        audio_dir1 = sys.argv[2]
        audio_dir2 = sys.argv[3]
        spec_dir1 = sys.argv[4]
        spec_dir2 = sys.argv[5]
        combine_dir = sys.argv[6] if len(sys.argv) > 6 else spec_dir1
        plotcombine(audio_dir1, audio_dir2, spec_dir1, spec_dir2, combine_dir)
    else:
        audio_dir = sys.argv[2]
        spec_dir = sys.argv[3] if len(sys.argv) > 3 else audio_dir
        plotstft(audio_dir, audio_name, savedir=spec_dir)


