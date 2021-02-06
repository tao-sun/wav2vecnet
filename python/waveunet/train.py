#!/usr/bin/python
import math
import os
import sys

from tqdm.contrib import tqdm

import h5py
import numpy as np

import torch
import torch.nn as nn
# import torchaudio
from torch.utils.data import DataLoader
# torchaudio.set_audio_backend("soundfile")
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
import torch.utils.data._utils.collate as collate
from prepare_data.timit_prepare import prepare_timit
from prepare_data.voicebank_prepare import prepare_voicebank
from prepare_data.hdf5_prepare import create_hdf5

from evaluate.quality_measures import SNRseg, composite

from fairseq.models.wav2vec import Wav2VecModel, Wav2Vec2Model


# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))



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


def compute_pesq(pred_wavs, target_wavs, lengths):
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


def compute_composite(pred_wavs, target_wavs, lengths):
    composites = Parallel(n_jobs=30)(
        delayed(composite)(
            fs=params.Sample_rate,
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


def compute_ssnr(pred_wavs, target_wavs, lengths):
    ssnrs = Parallel(n_jobs=30)(
        delayed(SNRseg)(
            fs=params.Sample_rate,
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


def pad_batch(batch_input):
    """
    Pad a patch so that they have the same number of frames
    """
    max_sample_frames = max([sample[0][1].shape[0] for sample in batch_input])
    new_batch = []

    for item in batch_input:
        name, noisy, audio_length = item[0]

        frame_num_diff = max_sample_frames - noisy.shape[0]
        padded_example = np.zeros((frame_num_diff, 1, noisy.shape[2]), dtype=noisy.dtype)
        new_noisy = np.concatenate([noisy, padded_example], 0)

        _, clean, clean_length = item[1]

        padded_clean = np.zeros((frame_num_diff, 1, noisy.shape[2]), dtype=clean.dtype)
        new_clean = np.concatenate([clean, padded_clean], 0)

        for i in range(max_sample_frames):
            new_batch.append(([name, new_noisy[i], audio_length], [name, new_clean[i], clean_length]))


    collated_batch = collate.default_collate(new_batch)
    # inputs, targets = collated_batch
    # print("inputs:" + str([sample for sample in inputs]))
    # print("targets:" + str([sample for sample in targets]))
    # print("batch0:" + str([sample for sample in collated_batch]))
    # print("batch1:" + str(collated_batch))
    return collated_batch


def ri_loss(pred, gt, N=512):
    pred_stft = torch.stft(pred, N)  # utils.stdft(pred, N)
    gt_stft = torch.stft(gt, N)  # utils.stdft(gt, N)

    r_pred_stft = pred_stft[:, :, :, 0]
    r_gt_stft = gt_stft[:, :, :, 0]
    r_loss = F.l1_loss(r_pred_stft, r_gt_stft)

    i_pred_stft = pred_stft[:, :, :, 1]
    i_gt_stft = gt_stft[:, :, :, 1]
    i_loss = F.l1_loss(i_pred_stft, i_gt_stft)

    loss = r_loss + i_loss
    return loss


def sm_loss(pred, gt, N=512):
    pred_stft = torch.stft(pred, N)  # utils.stdft(pred, N)
    gt_stft = torch.stft(gt, N)  # utils.stdft(gt, N)

    r_pred_stft = torch.abs(pred_stft[:, :, :, 0])
    i_pred_stft = torch.abs(pred_stft[:, :, :, 1])
    m_pred_stft = r_pred_stft + i_pred_stft

    r_gt_stft = torch.abs(gt_stft[:, :, :, 0])
    i_gt_stft = torch.abs(gt_stft[:, :, :, 1])
    m_gt_stft = r_gt_stft + i_gt_stft

    loss = F.l1_loss(m_pred_stft, m_gt_stft)
    return loss


def pcm_loss(inputs, pred, gt, N=512):
    pred_sm = sm_loss(pred, gt, N)

    pred_noise = inputs - pred
    gt_noise = inputs - gt
    noise_sm = sm_loss(pred_noise, gt_noise, N)

    pcm = 0.5 * pred_sm + 0.5 * noise_sm
    return pcm


if params.pretrained_model == "wav2vec":
    if params.wav2vec_version == 1.0:
        cp = torch.load(params.wav2vec1_model)
        wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
        wav2vec.load_state_dict(cp['model'])
        if params.use_device2 and torch.cuda.device_count() > 1:
            wav2vec.to(torch.device(params.device2))
        else:
            wav2vec.to(torch.device(params.device))
        for param in wav2vec.parameters():
            param.requires_grad = False
    elif params.wav2vec_version == 2.0:
        cp = torch.load(params.wav2vec2_model)
        wav2vec2 = Wav2Vec2Model.build_model(cp['args'])
        wav2vec2.load_state_dict(cp['model'])

        if params.use_device2 and torch.cuda.device_count() > 1:
            wav2vec2.to(torch.device(params.device2))
        else:
            wav2vec2.to(torch.device(params.device))

        for param in wav2vec2.parameters():
            param.requires_grad = False
elif params.pretrained_model == "dfl":
    featurenet = FeatureNet(2, [15, 7], [1, 2])
    featurenet.load_state_dict(torch.load(params.dfl_model))
    if params.use_device2 and torch.cuda.device_count() > 1:
        featurenet.to(torch.device(params.device2))
    else:
        featurenet.to(torch.device(params.device))
    for param in featurenet.parameters():
        param.requires_grad = False
else:
    raise Exception("Illegal 'pretrained_model' set in the .yaml file! Please choose from 'wav2vec' and 'dfl'.")


def compute_features(x):
    if params.use_device2 and torch.cuda.device_count() > 1:
        x = x.to(params.device2)
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
if params.dataset == "TIMIT":
    prepare_timit(
        data_folder=params.data_folder,
        save_folder=params.save_folder
    )
elif params.dataset == "voicebank":
    prepare_voicebank(
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
params.train_dataset2.set_shapes(params.model.shapes)
params.valid_dataset.set_shapes(params.model.shapes)
params.test_dataset.set_shapes(params.model.shapes)

if params.basic_loss == "MSE":
    train_set = DataLoader(params.train_dataset,
                           batch_size=params.N_batch,
                           shuffle=True,
                           num_workers=1,
                           worker_init_fn=worker_init_fn)
else:
    train_set = DataLoader(params.train_dataset2,
                           batch_size=params.N_batch,
                           shuffle=True,
                           collate_fn=pad_batch,
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
    def __init__(
            self,
            modules=None,
            optimizer=None,
            first_inputs=None,
            auto_mix_prec=False,
    ):
        super(SEBrain, self).__init__(modules, optimizer, first_inputs, auto_mix_prec)
        if params.pretrained_model == "wav2vec":
            self.feature_weights = torch.ones(len(params.wav2vec_loss_layers), device=params.device)
            self.weights_ratio = params.weights_ratio

    def compute_forward(self, x, stage="train", init_params=False):
        _, self.input_wavs, _ = x
        wavs = self.input_wavs.to(params.device)
        out = params.model(wavs)
        return out

    def compute_objectives(self, pred_wavs, targets, stage="train"):
        pred_wavs = torch.squeeze(pred_wavs, 1)
        pred_wavs = pred_wavs.to(params.device)

        name, target_wavs, length = targets
        target_wavs = target_wavs.to(params.device)
        start = params.model.shapes["output_start_frame"]
        end = start + pred_wavs.shape[1]
        target_wavs = torch.squeeze(target_wavs, 1)[:, start:end]

        input_wavs = self.input_wavs
        input_wavs = input_wavs.to(params.device)
        if stage != "train":
            input_wavs = input_wavs.contiguous().view(1, 1, -1)
        input_wavs = torch.squeeze(input_wavs, 1)[:, start:end]

        if params.basic_loss == "MSE":
            basic_loss = params.mse_cost(pred_wavs, target_wavs)
        else:
            batch_size = params.N_batch if stage == "train" else pred_wavs.shape[0]
            pred_wavs = pred_wavs.contiguous().view(batch_size, -1, pred_wavs.shape[1])
            pred_wavs = pred_wavs.contiguous().view(batch_size, -1)
            target_wavs = target_wavs.contiguous().view(batch_size, -1, target_wavs.shape[1])
            target_wavs = target_wavs.contiguous().view(batch_size, -1)
            input_wavs = input_wavs.contiguous().view(batch_size, -1, input_wavs.shape[1])
            input_wavs = input_wavs.contiguous().view(batch_size, -1)

            if params.basic_loss == "SM":
                basic_loss = sm_loss(pred_wavs, target_wavs)
            elif params.basic_loss == "RI":
                basic_loss = ri_loss(pred_wavs, target_wavs)
            elif params.basic_loss == "PCM":
                basic_loss = pcm_loss(input_wavs, pred_wavs, target_wavs)

        if params.combined_loss or stage != "train":
            predicted_features = compute_features(pred_wavs)
            target_features = compute_features(target_wavs)

            feature_losses = []
            total_feature_loss = 0.0
            for i in range(len(predicted_features)):
                feature_loss = params.mse_cost(predicted_features[i], target_features[i])
                feature_losses.append(feature_loss)
                if params.pretrained_model == "dfl":
                    total_feature_loss += feature_loss

            if params.pretrained_model == "wav2vec":
                self.effective_feature_losses = torch.zeros(len(params.wav2vec_loss_layers), device=params.device)
                for idx, layer_num in enumerate(params.wav2vec_loss_layers):
                    self.effective_feature_losses[idx] = feature_losses[layer_num]
                total_feature_loss = torch.sum(self.effective_feature_losses / self.feature_weights)

            if params.combined_loss:
                r1, r2 = self.weights_ratio.split(":")
                loss = float(r1) * basic_loss + float(r2) * total_feature_loss.to(params.device)
            else:
                loss = basic_loss
        else:
            loss = basic_loss

        stats = {}
        if stage != "train":
            pesq_scores = compute_pesq(
                pred_wavs.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
            )

            ssnrs = compute_ssnr(
                pred_wavs.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
            )

            stats["basic_loss"] = basic_loss

            stats["feature_loss"] = total_feature_loss
            for i, feature_loss in enumerate(feature_losses):
                stats["fl[{idx}]".format(idx=i)] = feature_loss

            stats['snr'] = [snr(pred_wavs, target_wavs, False)]
            stats["ssnrs"] = ssnrs
            # stats['snr_scaled'] = [snr(pred_wavs, target_wavs)]
            stats['si_sdr'] = [sisdr(pred_wavs, target_wavs)]
            stats["pesq"] = pesq_scores
            if stage == "test":
                csigs, cbaks, covls, ssnrs = compute_composite(
                    pred_wavs.cpu().numpy(),
                    target_wavs.cpu().numpy(),
                    np.array([length]),
                )
                stats["csigs"] = csigs
                stats["cbaks"] = cbaks
                stats["covls"] = covls

            stats["stoi"] = -stoi_loss(pred_wavs, target_wavs, torch.Tensor([length])).unsqueeze(0)

            if stage == "test":
                enhance_path = os.path.join(params.enhanced_folder, name + ".wav")
                sf.write(enhance_path, pred_wavs.cpu().numpy().squeeze(), params.Sample_rate)
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

        _, self.input_wavs, audio_length = inputs
        input_wavs = self.input_wavs.to(params.device)
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

        # if params.pretrained_model == "wav2vec" and epoch == params.set_weights_epoch:
        #     self.weights_ratio = params.weights_ratio
        #     self.feature_weights = torch.div(self.effective_feature_losses, torch.sum(self.effective_feature_losses))
        #     print("feature weights: %s" % self.feature_weights)

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
