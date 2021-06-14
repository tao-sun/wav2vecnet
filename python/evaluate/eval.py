import torch
import math
import sys, os

from tqdm import tqdm
# from scipy.io import wavfile
import soundfile as sf

from joblib import Parallel, delayed
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from evaluate.quality_measures import SNRseg, composite
from pesq import pesq

import numpy as np

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from prepare_data.hdf5_prepare import get_samples  # noqa E402


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


def add_stats(dataset_stats, batch_stats):
    """Add the stats for a batch to the set of stats for a dataset.

    Arguments
    ---------
    dataset_stats : dict
        A mapping of stat name to a list of the stats in the dataset.
    batch_stats : dict
        A mapping of stat name to the value for that stat in a batch.
    """
    for key in batch_stats:
        if key not in dataset_stats:
            dataset_stats[key] = []
        if isinstance(batch_stats[key], list):
            dataset_stats[key].extend(batch_stats[key])
        else:
            dataset_stats[key].append(batch_stats[key])


def summarize_average(stat_list):
    return float(sum(stat_list) / len(stat_list))


def batch_stats(input_wavs, target_wavs, lens, sample_rate, device=None):
    if device is not None:
        input_wavs = input_wavs.to(torch.device(device))
        target_wavs = target_wavs.to(torch.device(device))
        lens = lens.to(torch.device(device))

    pesq_scores = compute_pesq(
        input_wavs.cpu().numpy(),
        target_wavs.cpu().numpy(),
        np.array([lens]),
        sr
    )

    ssnrs = compute_ssnr(
        input_wavs.cpu().numpy(),
        target_wavs.cpu().numpy(),
        np.array([lens]),
        sr
    )


    batch_stats = {}
    batch_stats["snr"] = snr(input_wavs, target_wavs, False)
    batch_stats["ssnrs"] = ssnrs
    batch_stats['si_sdr'] = [sisdr(input_wavs, target_wavs)]
    batch_stats["pesq"] = pesq_scores

    csigs, cbaks, covls = compute_composite(
        input_wavs.cpu().numpy(),
        target_wavs.cpu().numpy(),
        np.array([lens]),
        sr
    )
    batch_stats["csigs"] = csigs
    batch_stats["cbaks"] = cbaks
    batch_stats["covls"] = covls

    batch_stats["stoi"] = -stoi_loss(input_wavs, target_wavs, torch.Tensor([lens])).unsqueeze(0)

    return batch_stats


"python -u python/eval.py ./results/waveunet/TIMIT_mse/save/test.csv ./results/waveunet/TIMIT_mse/enhanced 3"
if __name__ == '__main__':
    csv_file = sys.argv[1]
    enhanced_path = sys.argv[2]
    snr_level = sys.argv[3] if len(sys.argv) > 3 else "all"
    batch_size = sys.argv[4] if len(sys.argv) > 4 else 1
    device = sys.argv[5] if len(sys.argv) > 5 else 'cuda:0'

    noisy_wavs = []
    clean_wavs = []
    enhanced_wavs = []
    lengths = []
    batch_empty = True

    samples = get_samples(csv_file, snr_level)
    dataset_stats = {}
    for idx, example in enumerate(tqdm(samples)):
        enhanced_name = example["ID"]
        # Load mix
        noisy_audio, _ = sf.read(os.path.join(enhanced_path, enhanced_name + "_noisy.wav"))  # wavfile.read(example["noisy_wav"])
        clean_audio, sr = sf.read(os.path.join(enhanced_path, enhanced_name + "_clean.wav"))  # wavfile.read(example["clean_wav"])
        enhanced_audio, _ = sf.read(os.path.join(enhanced_path, enhanced_name + ".wav"))  # wavfile.read(os.path.join(enhanced_path, enhanced_name + ".wav"))
        # noisy_audio, sr = sf.read(example["noisy_wav"])
        # clean_audio, _ = sf.read(example["clean_wav"])
        # enhanced_audio, _ = sf.read(example["noisy_wav"])

        noisy_wavs.append(noisy_audio)
        clean_wavs.append(clean_audio)
        enhanced_wavs.append(enhanced_audio)
        lengths.append(len(noisy_audio))
        batch_empty = False

        if len(noisy_wavs) == batch_size:
            noisy_wavs = torch.tensor(noisy_wavs)
            clean_wavs = torch.tensor(clean_wavs)
            enhanced_wavs = torch.tensor(enhanced_wavs)
            lengths = torch.tensor(lengths)
            stats = batch_stats(enhanced_wavs, clean_wavs, lengths, sr, device=device)
            add_stats(dataset_stats, stats)

            noisy_wavs = []
            clean_wavs = []
            enhanced_wavs = []
            lengths = []
            batch_empty = True

    if batch_empty is False:
        noisy_wavs = torch.tensor(noisy_wavs)
        clean_wavs = torch.tensor(clean_wavs)
        enhanced_wavs = torch.tensor(enhanced_wavs)
        lengths = torch.tensor(lengths)
        stats = batch_stats(enhanced_wavs, clean_wavs, lengths, device=device)
        add_stats(dataset_stats, stats)

    summary = {}
    for stat, value_list in dataset_stats.items():
        summary[stat] = summarize_average(value_list)
    print(summary)
