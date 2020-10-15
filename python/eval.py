import torch
import math
import sys, os

from tqdm import tqdm
# from scipy.io import wavfile
import soundfile as sf

from joblib import Parallel, delayed
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from pesq import pesq

import numpy as np

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import get_samples  # noqa E402


def snr(pred_wavs, target_wavs, scale=True):
    def rms(wavs):
        return torch.sqrt(torch.mean(torch.pow(wavs, 2)))

    if scale:
        target_wavs = target_wavs - torch.mean(target_wavs, 1, True).expand_as(target_wavs)
        pred_wavs = pred_wavs - torch.mean(pred_wavs, 1, True).expand_as(pred_wavs)

        target_wavs_max = torch.max(torch.abs(target_wavs), 1, True)[0]
        pred_wavs_max = torch.max(torch.abs(pred_wavs), 1, True)[0]
        pred_wavs = pred_wavs * (target_wavs_max / pred_wavs_max).expand_as(pred_wavs)

    noise_wavs = pred_wavs.float() - target_wavs.float()
    rms_signal = rms(target_wavs.float())
    rms_noise = rms(noise_wavs)

    snr_db = 20 * torch.log(rms_signal/rms_noise) / math.log(10.)
    return snr_db


def multiprocess_evaluation(pred_wavs, target_wavs, lengths, sample_rate):
    pesq_scores = Parallel(n_jobs=30)(
        delayed(pesq)(
            fs=sample_rate,
            ref=clean[: int(length)],
            deg=enhanced[: int(length)],
            mode="wb",
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    return pesq_scores


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

    batch_stats = {}
    batch_stats["snr"] = snr(input_wavs, target_wavs, False)
    batch_stats["snr_scaled"] = snr(input_wavs, target_wavs)
    batch_stats["pesq"] = multiprocess_evaluation(
        input_wavs.cpu().numpy(),
        target_wavs.cpu().numpy(),
        np.array([lengths]),
        sample_rate
    )
    batch_stats["stoi"] = -stoi_loss(input_wavs, target_wavs, torch.Tensor([lens])).unsqueeze(0)

    return batch_stats


if __name__ == '__main__':
    csv_file = sys.argv[1]
    enhanced_path = sys.argv[2]
    snr_level = sys.argv[3]
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
        noisy_audio, _ = sf.read(example["noisy_wav"])  # wavfile.read(example["noisy_wav"])
        clean_audio, sr = sf.read(example["clean_wav"])  # wavfile.read(example["clean_wav"])
        enhanced_audio, _ = sf.read(os.path.join(enhanced_path, enhanced_name + ".wav"))  # wavfile.read(os.path.join(enhanced_path, enhanced_name + ".wav"))

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
