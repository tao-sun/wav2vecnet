#!/usr/bin/python
import math
import os
import sys

from tqdm.contrib import tqdm

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

from dfl.network import FeatureNet
import torch.nn.functional as F

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


if params.pretrained_model == "wav2vec":
    if params.wav2vec_version == 1.0:
        cp = torch.load(params.wav2vec1_model)
        wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
        wav2vec.load_state_dict(cp['model'])
        wav2vec.to(torch.device(params.device))
        for param in wav2vec.parameters():
            param.requires_grad = False
    elif params.wav2vec_version == 2.0:
        cp = torch.load(params.wav2vec2_model)
        wav2vec2 = Wav2Vec2Model.build_model(cp['args'])
        wav2vec2.load_state_dict(cp['model'])
        wav2vec2.to(torch.device(params.device))
        for param in wav2vec2.parameters():
            param.requires_grad = False
elif params.pretrained_model == "dfl":
    featurenet = FeatureNet(2, [15, 7], [1, 2])
    featurenet.load_state_dict(torch.load(params.dfl_model))
    featurenet.to(torch.device(params.device))
    for param in featurenet.parameters():
        param.requires_grad = False
else:
    raise Exception("Illegal 'pretrained_model' set in the .yaml file! Please choose from 'wav2vec' and 'dfl'.")


def compute_features(x):
    if params.pretrained_model == "wav2vec":
        if params.wav2vec_version == 1.0:
            return compute_features1(x)
        elif params.wav2vec_version == 2.0:
            return compute_features2(x)
    elif params.pretrained_model == "dfl":
        return compute_features_dfl(x)


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


def compute_features_dfl(wavs):
    features = []

    wavs = wavs.view(wavs.shape[0], 1, wavs.shape[1])
    for i in range(1, params.dfl_layers + 1):
        conv_layer = getattr(featurenet, "conv" + str(i))
        norm_layer = getattr(featurenet, "batnorm" + str(i))

        wavs = conv_layer(wavs)
        wavs = F.leaky_relu(norm_layer(wavs))
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
params.valid_dataset.set_shapes(params.model.shapes)
params.test_dataset.set_shapes(params.model.shapes)
train_set = DataLoader(params.train_dataset,
                       batch_size=params.N_batch,
                       shuffle=True,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)
valid_set = DataLoader(params.valid_dataset,
                       batch_size=None,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)
test_set = DataLoader(params.test_dataset,
                       batch_size=None,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn)


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        _, wavs, _ = x
        wavs = wavs.to(params.device)
        out = params.model(wavs)
        return out

    def compute_objectives(self, predictions, targets, stage="train"):
        predictions = torch.squeeze(predictions, 1)

        name, target_wavs, length = targets
        start = params.model.shapes["output_start_frame"]
        end = start + predictions.shape[1]
        target_wavs = torch.squeeze(target_wavs, 1)[:, start:end]

        target_wavs = target_wavs.to(params.device)
        mse_loss = params.compute_cost(predictions, target_wavs)

        predicted_features = compute_features(predictions)
        target_features = compute_features(target_wavs)
        total_feature_loss = 0
        feature_losses = []
        for i in range(len(predicted_features)):
            feature_loss = params.compute_cost(predicted_features[i], target_features[i])
            feature_losses.append(feature_loss)
            if params.pretrained_model == "wav2vec":
                if i in params.wav2vec_loss_layers:
                    total_feature_loss += feature_loss
            elif params.pretrained_model == "dfl":
                total_feature_loss += feature_loss

        # loss = 0.995 * mse_loss + 0.005 * total_feature_loss
        loss = mse_loss

        stats = {}
        if stage != "train":
            pesq_scores = multiprocess_evaluation(
                predictions.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
            )

            stats["mse_loss"] = mse_loss
            stats["feature_loss"] = total_feature_loss
            for i, feature_loss in enumerate(feature_losses):
                stats["fl[{idx}]".format(idx=i)] = feature_loss
            stats['snr'] = [snr(predictions, target_wavs, False)]
            stats['snr_scaled'] = [snr(predictions, target_wavs)]
            stats['si_sdr'] = [sisdr(predictions, target_wavs)]
            stats["pesq"] = pesq_scores
            stats["stoi"] = -stoi_loss(predictions, target_wavs, torch.Tensor([length])).unsqueeze(0)

            if stage == "test":
                enhance_path = os.path.join(params.enhanced_folder, name + ".wav")
                sf.write(enhance_path, predictions.cpu().numpy().squeeze(), params.Sample_rate)
                clean_path = os.path.join(params.enhanced_folder, name + "_clean.wav")
                sf.write(clean_path, target_wavs.cpu().numpy().squeeze(), params.Sample_rate)

        return loss, stats

    def evaluate_batch(self, batch, stage="test"):
        """Evaluate one batch, override for different procedure than train.

        The default impementation depends on two methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : str
            The stage of the training process, one of "valid", "test"

        Returns
        -------
        dict
            A dictionary of the same format as ``fit_batch()`` where each item
            includes a statistic about the batch, including the loss.
            (e.g. ``{"loss": 0.1, "accuracy": 0.9}``)
        """
        inputs, targets = batch

        _, input_wavs, audio_length = inputs
        input_wavs = input_wavs.to(params.device)
        pred_wavs = params.model(input_wavs)
        pred_wavs = pred_wavs.view(1, 1, pred_wavs.shape[0] * pred_wavs.shape[2])[:, :, :audio_length]

        name, clean, clean_length = targets
        clean = clean.to(params.device)
        clean = torch.unsqueeze(clean, 1)
        targets = [name, clean, clean_length]

        loss, stats = self.compute_objectives(pred_wavs, targets, stage=stage)
        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
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

        if epoch % 5 == 0:
            # Load best checkpoint for evaluation
            # params.checkpointer.recover_if_possible(max_key="pesq_score")
            test_stats = self.evaluate(test_set)
            params.train_logger.log_stats(
                stats_meta={"Epoch loaded": params.epoch_counter.current},
                test_stats=test_stats,
            )
            # params.checkpointer.recover_if_possible()


params.model.to(torch.device(params.device))

first_x, first_y = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, train_set, valid_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(max_key="pesq_score")
test_stats = se_brain.evaluate(test_set)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
