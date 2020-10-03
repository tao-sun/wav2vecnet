#!/usr/bin/python
import math
import os
import sys

import h5py
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader
torchaudio.set_audio_backend("soundfile")
from joblib import Parallel, delayed
from pesq import pesq
from torch import FloatTensor, LongTensor

import speechbrain as sb
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.train_logger import summarize_average

import soundfile as sf

from utils import worker_init_fn

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import prepare_timit, create_hdf5  # noqa E402

from fairseq.models.wav2vec import Wav2VecModel, Wav2Vec2Model



# Load hyperparameters file with command-line overrides
params_file, overrides = sb.core.parse_arguments(sys.argv[1:])
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin, overrides)

# Create experiment directory
sb.core.create_experiment_directory(
    experiment_directory=params.output_folder,
    hyperparams_to_save=params_file,
    overrides=overrides,
)

if params.use_tensorboard:
    from speechbrain.utils.train_logger import TensorboardLogger

    tensorboard_train_logger = TensorboardLogger(params.tensorboard_logs)

# Create the folder to save enhanced files
if not os.path.exists(params.enhanced_folder):
    os.mkdir(params.enhanced_folder)


def truncate(wavs, lengths, max_length):
    lengths *= max_length / wavs.shape[1]
    lengths = lengths.clamp(max=1)
    wavs = wavs[:, :max_length]
    return wavs, lengths


def multiprocess_evaluation(pred_wavs, target_wavs, lengths):
    pesq_scores = Parallel(n_jobs=30)(
        delayed(pesq)(
            fs=params.Sample_rate,
            ref=clean[: int(length)],
            deg=enhanced[: int(length)],
            mode="wb",
        )
        for enhanced, clean, length in zip(pred_wavs, target_wavs, lengths)
    )
    return pesq_scores


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


def evaluate(hdf_file, model, write=False):
    stats = {}
    with h5py.File(hdf_file, "r") as f:
        for idx in range(len(f)):
            audio_length = f[str(idx)].attrs["length"]
            clean_length = f[str(idx)].attrs["clean_length"]

            audio = f[str(idx)]["noisy"]
            clean = f[str(idx)]["clean"]
            enhanced = evaluate_audio(audio, model)

            pesq_scores = multiprocess_evaluation(
                enhanced.cpu().numpy(),
                clean.cpu().numpy(),
                clean_length.cpu().numpy(),
            )
            stats['snr'] = snr(enhanced, clean, False)
            stats['snr_scaled'] = snr(enhanced, clean)
            stats["pesq"] = pesq_scores
            stats["stoi"] = -stoi_loss(enhanced, clean, clean_length)


            if write is True:
                name = f[str(idx)].attrs["ID"]
                enhance_path = os.path.join(params.enhanced_folder, name)
                sf.write(enhance_path, enhanced.to("cpu"), 16000)

    return stats


def evaluate_audio(audio, model):
    output_shift = model.shapes["output_frames"]

    pad_back = audio.shape[1] % output_shift
    pad_back = 0 if pad_back == 0 else output_shift - pad_back
    if pad_back > 0:
        audio = np.pad(audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)

    target_outputs = audio.shape[1]
    output = np.zeros(audio.shape, np.float32)

    # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
    pad_front_context = model.shapes["output_start_frame"]
    pad_back_context = model.shapes["input_frames"] - model.shapes["output_end_frame"]
    audio = np.pad(audio, [(0, 0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

    # Iterate over mixture magnitudes, fetch network prediction
    with torch.no_grad():
        for target_start_pos in range(0, target_outputs, model.shapes["output_frames"]):

            # Prepare mixture excerpt by selecting time interval
            curr_input = audio[:, target_start_pos:target_start_pos + model.shapes[
                "input_frames"]]  # Since audio was front-padded input of [targetpos:targetpos+inputframes] actually predicts [targetpos:targetpos+outputframes] target range

            # Convert to Pytorch tensor for model prediction
            curr_input = torch.from_numpy(curr_input).unsqueeze(0)

            # Predict
            curr_target = model(curr_input).to(torch.device(params.device))
            output[:, target_start_pos:target_start_pos + model.shapes["output_frames"]] = curr_target.squeeze(0).cpu().numpy()

    return torch.from_numpy(output).to(torch.device(params.device))


if params.wav2vec_version == 1.0:
    cp = torch.load(params.wav2vec1_model)
    wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec.load_state_dict(cp['model'])
    wav2vec.to(torch.device(params.device))
elif params.wav2vec_version == 2.0:
    cp = torch.load(params.wav2vec2_model)
    wav2vec2 = Wav2Vec2Model.build_model(cp['args'])
    wav2vec2.load_state_dict(cp['model'])
    wav2vec2.to(torch.device(params.device))


def compute_features(x):
    if params.wav2vec_version == 1.0:
        return compute_features1(x)
    elif params.wav2vec_version == 2.0:
        return compute_features2(x)


def compute_features1(wavs):
    feature_extractor = wav2vec.feature_extractor

    features = []
    wavs = wavs.unsqueeze(1)
    for i, conv in enumerate(feature_extractor.conv_layers):
        residual = wavs
        wavs = conv(wavs)
        if feature_extractor.skip_connections and wavs.size(1) == residual.size(1):
            tsz = wavs.size(2)
            r_tsz = residual.size(2)
            residual = residual[..., :: r_tsz // tsz][..., :tsz]
            wavs = (wavs + residual) * feature_extractor.residual_scale
        features.append(wavs)

    return features


def compute_features2(wavs):
    feature_extractor = wav2vec2.feature_extractor

    features = []
    wavs = wavs.unsqueeze(1)
    for i, conv in enumerate(feature_extractor.conv_layers):
        wavs = conv(wavs)
        features.append(wavs)

    return features

# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    save_folder=params.save_folder
)

if not os.path.exists(params.hdf5_train):
    create_hdf5(params.csv_train, params.hdf5_train, params.Sample_rate)
if not os.path.exists(params.hdf5_valid):
    create_hdf5(params.csv_valid, params.hdf5_valid, params.Sample_rate)
if not os.path.exists(params.hdf5_test):
    create_hdf5(params.csv_test, params.hdf5_test, params.Sample_rate)

params.train_dataset.set_shapes(params.model.shapes)
train_set = DataLoader(params.train_dataset,
                       batch_size=params.N_batch,
                       shuffle=True,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)
params.valid_dataset.set_shapes(params.model.shapes)
valid_set = DataLoader(params.valid_dataset,
                       batch_size=params.N_batch,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)
params.test_dataset.set_shapes(params.model.shapes)
test_set = DataLoader(params.test_dataset,
                       batch_size=params.N_batch,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)

class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        # ids, wavs, lens = x
        # wavs, lens = truncate(wavs, lens, params.max_length)
        # wavs = torch.unsqueeze(wavs, -1)
        # wavs, lens = wavs.to(params.device), lens.to(params.device)
        wavs = x.to(params.device)
        out = params.model(wavs)
        return out

    def compute_objectives(self, predictions, targets, stage="train"):
        # ids, target_wavs, lens = targets
        # target_wavs, lens = truncate(target_wavs, lens, params.max_length)
        predictions = torch.squeeze(predictions, -2)
        targets = torch.squeeze(targets, -2)[:, params.model.shapes["output_start_frame"]:params.model.shapes["output_start_frame"]+params.model.shapes["output_frames"]]
        target_wavs = targets.to(params.device)
        # lens = lens.to(params.device)
        mse_loss = params.compute_cost(predictions, target_wavs)

        predicted_features = compute_features(predictions)
        target_features = compute_features(target_wavs)
        feature_loss = 0
        for i in range(len(predicted_features)):
            feature_loss += params.compute_cost(predicted_features[i], target_features[i])

        loss = mse_loss

        stats = {}
        return loss, stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        valid_stats = evaluate(params.hdf5_valid, params.model)

        if params.use_tensorboard:
            tensorboard_train_logger.log_stats(
                {"Epoch": epoch}, train_stats, valid_stats
            )

        params.train_logger.log_stats(
            {"Epoch": epoch}, train_stats, valid_stats
        )


        pesq_score = summarize_average(valid_stats["pesq"])
        params.checkpointer.save_and_keep_only(
            meta={"pesq_score": pesq_score}, max_keys=["pesq_score"],
        )

        if epoch % 2 == 0:
            # Load best checkpoint for evaluation
            params.checkpointer.recover_if_possible(max_key="pesq_score")
            test_stats = evaluate(params.hdf5_test, params.model, write=True)
            params.train_logger.log_stats(
                stats_meta={"Epoch loaded": params.epoch_counter.current},
                test_stats=test_stats,
            )
            params.checkpointer.recover_if_possible()


params.model.to(torch.device(params.device))

first_x, first_y = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(max_key="pesq_score")
test_stats = evaluate(params.hdf5_test, params.model, write=True)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
