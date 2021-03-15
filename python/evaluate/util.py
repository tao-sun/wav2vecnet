import numpy as np
from scipy.signal import firls,kaiser,upfirdn
from fractions import Fraction

import torch
import math
from joblib import Parallel, delayed
from evaluate.quality_measures import SNRseg, composite
from pesq import pesq


def resample_matlab_like(x_orig,p,q):
    if len(x_orig.shape)>2:
        raise ValueError('x must be a vector or 2d matrix')
        
    if x_orig.shape[0]<x_orig.shape[1]:
        x=x_orig.T
    else:
        x= x_orig
    beta = 5
    N = 10
    frac=Fraction(p, q)
    p = frac.numerator
    q = frac.denominator
    pqmax = max(p,q)
    fc = 1/2/pqmax
    L = 2*N*pqmax + 1
    h = firls( L, np.array([0,2*fc,2*fc,1]), np.array([1,1,0,0]))*kaiser(L,beta)
    h = p*h/sum(h)

    Lhalf = (L-1)/2
    Lx = x.shape[0]

    nz = int(np.floor(q-np.mod(Lhalf,q)))
    z = np.zeros((nz,))
    h = np.concatenate((z,h))
    Lhalf = Lhalf + nz
    delay = int(np.floor(np.ceil(Lhalf)/q))
    nz1 = 0
    while np.ceil( ((Lx-1)*p+len(h)+nz1 )/q ) - delay < np.ceil(Lx*p/q):
        nz1 = nz1+1
    h = np.concatenate((h,np.zeros(nz1,)))
    y = upfirdn(h,x,p,q,axis=0)
    Ly = int(np.ceil(Lx*p/q))
    y = y[delay:]
    y = y[:Ly]
    
    if x_orig.shape[0]<x_orig.shape[1]:
        y=y.T
    
    return y


def compute_pesq(pred_wavs, target_wavs, lengths, sr):
    pesq_scores = Parallel(n_jobs=30)(
        delayed(pesq)(
            fs=sr,
            ref=clean[: int(length)],
            deg=enhanced[: int(length)],
            mode="wb",
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    return pesq_scores


def compute_composite(pred_wavs, target_wavs, lengths, sr):
    composites = Parallel(n_jobs=30)(
        delayed(composite)(
            fs=sr,
            clean_speech=clean[: int(length)],
            processed_speech=enhanced[: int(length)]
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    csigs, cbaks, covls =[], [], []
    for csig, cbak, covl in composites:
        csigs.append(csig)
        cbaks.append(cbak)
        covls.append(covl)

    return csigs, cbaks, covls


def compute_ssnr(pred_wavs, target_wavs, lengths, sr):
    ssnrs = Parallel(n_jobs=30)(
        delayed(SNRseg)(
            fs=sr,
            clean_speech=clean[: int(length)],
            processed_speech=enhanced[: int(length)]
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    return ssnrs


def snr(pred_wavs, target_wavs, scale=True):
    def rms(wavs):
        return math.sqrt(torch.mean(torch.pow(wavs, 2)))

    if scale:
        target_wavs = target_wavs - torch.mean(target_wavs, 1, True).expand_as(target_wavs)
        pred_wavs = pred_wavs - torch.mean(pred_wavs, 1, True).expand_as(pred_wavs)

        target_wavs_max = torch.max(torch.abs(target_wavs), 1, True)[0]
        pred_wavs_max = torch.max(torch.abs(pred_wavs), 1, True)[0]
        pred_wavs = pred_wavs * (target_wavs_max / pred_wavs_max).expand_as(pred_wavs)

    noise_wavs = pred_wavs.float() - target_wavs.float()
    rms_signal = rms(target_wavs.float())
    rms_noise = rms(noise_wavs)

    snr_db = 20 * math.log(rms_signal/rms_noise) / math.log(10.)
    return snr_db


def sisdr(pred_wavs, target_wavs, zero_mean=True):
    EPS = 1e-8

    if target_wavs.size() != pred_wavs.size() or target_wavs.ndim != 2:
        raise TypeError(
            f"Inputs must be of shape [batch, time], got {target_wavs.size()} and {pred_wavs.size()} instead"
        )
    # Step 1. Zero-mean norm
    if zero_mean:
        mean_source = torch.mean(target_wavs, dim=1, keepdim=True)
        mean_estimate = torch.mean(pred_wavs, dim=1, keepdim=True)
        target_wavs = target_wavs - mean_source
        pred_wavs = pred_wavs - mean_estimate

    # [batch, 1]
    dot = torch.sum(pred_wavs * target_wavs, dim=1, keepdim=True)
    # [batch, 1]
    s_target_energy = torch.sum(target_wavs ** 2, dim=1, keepdim=True) + EPS
    # [batch, time]
    scaled_target = dot * target_wavs / s_target_energy
    e_noise = pred_wavs - scaled_target
    # [batch]
    sisdr = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
    sisdr = 10 * torch.log10(sisdr + EPS)
    sisdr = sisdr.mean()
    return sisdr
