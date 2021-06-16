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

import torch.nn.functional as F
import torch.utils.data._utils.collate as collate
from prepare_data.timit_prepare import prepare_timit
from prepare_data.voicebank_prepare import prepare_voicebank
from prepare_data.hdf5_prepare import create_hdf5

from evaluate.util import compute_pesq, compute_composite, compute_ssnr, sisdr, snr

from feature_loss_factory import FeatureLossFactory


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


def sisnr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)

    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


if params.pretrained_model == "wav2vec":
    if params.wav2vec_version == 1.0:
        cp = torch.load(params.wav2vec1_model)
        pretrained = Wav2VecModel.build_model(cp['args'], task=None)
        pretrained.load_state_dict(cp['model'])
    elif params.wav2vec_version == 2.0:
        cp_path = params.wav2vec2_model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        pretrained = model[0]
        # cp = torch.load(params.wav2vec2_model)
        # wav2vec = Wav2Vec2Model.build_model(cp['args'], task=None)
        # wav2vec.load_state_dict(cp['model'])
    elif params.wav2vec_version == "xlsr":
        cp_path = params.xlsr_model
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        pretrained = model[0]
elif params.pretrained_model == "dfl":
    pretrained = FeatureNet(2, [15, 7], [1, 2])
    pretrained.load_state_dict(torch.load(params.dfl_model))
elif params.pretrained_model == "pase":
    from pase.models.frontend import wf_builder
    pretrained = wf_builder(params.pase_cfg).eval()
    pretrained.load_pretrained(params.pase_model, load_last=True, verbose=True)
elif params.pretrained_model == "rawnet":
    from rawnet.model_RawNet2_original_code import RawNet
    from rawnet.parser import get_args
    args = get_args()
    args.model['nb_classes'] = 6112
    pretrained = RawNet(args.model)
    pretrained.load_state_dict(torch.load(params.rawnet_model))
elif params.pretrained_model == "audioset_tagging_cnn":
    from audioset_tagging_cnn import parser, config
    from audioset_tagging_cnn.models import Cnn14, Wavegram_Cnn14
    args = parser.get_args()
    pretrained = Cnn14(sample_rate=args.sample_rate, window_size=args.window_size,
                       hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                       classes_num=config.classes_num)

    checkpoint = torch.load(params.audioset_tagging_cnn_model)
    pretrained.load_state_dict(checkpoint['model'])
else:
    raise Exception("Illegal 'pretrained_model' set in the .yaml file! Please choose from 'wav2vec' and 'dfl'.")

if params.use_device2 and torch.cuda.device_count() > 1:
    pretrained.to(torch.device(params.device2))
else:
    pretrained.to(torch.device(params.device))
for param in pretrained.parameters():
    param.requires_grad = False
pretrained.eval()


def compute_features(x):
    if params.use_device2 and torch.cuda.device_count() > 1:
        x = x.to(params.device2)
    if params.pretrained_model == "wav2vec":
        if params.wav2vec_version == 1.0:
            return compute_features1(x, pretrained)
        elif (params.wav2vec_version == 2.0) or (params.wav2vec_version == "xlsr"):
            return compute_features2(x, pretrained)
    elif params.pretrained_model == "dfl":
        return compute_features_dfl(x, pretrained)
    elif params.pretrained_model == "pase":
        return compute_features_pase(x, pretrained)
    elif params.pretrained_model == "rawnet":
        return compute_features_rawnet(x, pretrained)
    elif params.pretrained_model == "audioset_tagging_cnn":
        return compute_features_audioset_tagging_cnn(x, pretrained)


def compute_features1(wavs, wav2vec):
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


def compute_features2(wavs, wav2vec2):
    feature_extractor = wav2vec2.feature_extractor

    features = []
    wavs = wavs.unsqueeze(1)
    for i, conv in enumerate(feature_extractor.conv_layers):
        wavs = conv(wavs)
        features.append(wavs)

    return features


def compute_features_dfl(wavs, featurenet):
    features = []

    wavs = wavs.view(wavs.shape[0], 1, wavs.shape[1])
    for i in range(1, params.dfl_layers + 1):
        conv_layer = getattr(featurenet, "conv" + str(i))
        norm_layer = getattr(featurenet, "batnorm" + str(i))

        wavs = conv_layer(wavs)
        wavs = F.leaky_relu(norm_layer(wavs))
        features.append(wavs)
    return features


def compute_features_pase(wavs, pase_model):
    wavs = pase_model(torch.unsqueeze(wavs, 1))
    return [wavs]


def compute_features_rawnet(wavs, rawnet):
    features = []

    d_len_seq = rawnet.len_seq
    nb_samp = wavs.shape[0]
    len_seq = wavs.shape[1]

    if len_seq < d_len_seq:
        wavs = F.pad(wavs, (0, d_len_seq - len_seq), "constant", 0)
    else:
        wavs = wavs[:, :d_len_seq]
    len_seq = d_len_seq

    x = rawnet.ln(wavs)
    x = x.view(nb_samp, 1, len_seq)
    x = F.max_pool1d(torch.abs(rawnet.first_conv(x)), 3)
    x = rawnet.first_bn(x)
    x = rawnet.lrelu_keras(x)
    features.append(x)

    x0 = rawnet.block0(x)
    y0 = rawnet.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
    y0 = rawnet.fc_attention0(y0)
    y0 = rawnet.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
    x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)
    features.append(x)

    x1 = rawnet.block1(x)
    y1 = rawnet.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
    y1 = rawnet.fc_attention1(y1)
    y1 = rawnet.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
    x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)
    features.append(x)

    return features


def compute_features_audioset_tagging_cnn(wavs, audioset_tagging_cnn):
    features = []

    x = audioset_tagging_cnn.spectrogram_extractor(wavs)  # (batch_size, 1, time_steps, freq_bins)
    x = audioset_tagging_cnn.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

    x = x.transpose(1, 3)
    x = audioset_tagging_cnn.bn0(x)
    x = x.transpose(1, 3)

    if audioset_tagging_cnn.training:
        x = audioset_tagging_cnn.spec_augmenter(x)

    x = audioset_tagging_cnn.conv_block1(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=audioset_tagging_cnn.training)
    features.append(x)

    x = audioset_tagging_cnn.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=audioset_tagging_cnn.training)
    features.append(x)

    x = audioset_tagging_cnn.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=audioset_tagging_cnn.training)
    features.append(x)

    x = audioset_tagging_cnn.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=audioset_tagging_cnn.training)
    features.append(x)

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

if params.basic_loss == ("MSE" or "SISDR"):
    train_set = DataLoader(params.train_dataset,
                           batch_size=params.N_batch,
                           shuffle=True,
                           num_workers=1,
                           worker_init_fn=worker_init_fn,
                           pin_memory=True)
else:
    train_set = DataLoader(params.train_dataset2,
                           batch_size=params.N_batch,
                           shuffle=True,
                           collate_fn=pad_batch,
                           num_workers=1,
                           worker_init_fn=worker_init_fn,
                           drop_last=True)
valid_set = DataLoader(params.valid_dataset,
                       batch_size=None,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn,
                       pin_memory=True)
test_set = DataLoader(params.test_dataset,
                       batch_size=None,
                       shuffle=False,
                       num_workers=1,
                       worker_init_fn=worker_init_fn,
                       pin_memory=True)


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

    def compute_forward(self, batch, stage="train", init_params=False):
        inputs, targets = batch

        _, input_wavs, _ = inputs
        wavs = input_wavs.to(params.device)
        out = params.model(wavs)
        return out

    def compute_objectives(self, pred_wavs, batch, stage="train"):
        inputs, targets = batch

        pred_wavs = torch.squeeze(pred_wavs, 1)
        pred_wavs = pred_wavs.to(params.device)

        name, target_wavs, length = targets
        target_wavs = target_wavs.to(params.device)
        start = params.model.shapes["output_start_frame"]
        end = start + pred_wavs.shape[1]
        target_wavs = torch.squeeze(target_wavs, 1)[:, start:end]

        _, input_wavs, _ = inputs
        input_wavs = input_wavs.to(params.device)
        if stage != "train":
            input_wavs = input_wavs.contiguous().view(1, 1, -1)
        input_wavs = torch.squeeze(input_wavs, 1)[:, start:end]

        if params.basic_loss == "MSE":
            basic_loss = params.mse_cost(pred_wavs, target_wavs)
        elif params.basic_loss == "SISDR":
            sisdr_val = sisnr(pred_wavs, target_wavs)
            basic_loss = -torch.mean(sisdr_val)
        else:
            batch_size = params.N_batch if stage == "train" else pred_wavs.shape[0]

            if (pred_wavs.shape[0] * pred_wavs.shape[1]) % (batch_size * pred_wavs.shape[1]) != 0:
                print(pred_wavs.shape)
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
                if params.pretrained_model != "wav2vec":
                    total_feature_loss += feature_loss

            if params.pretrained_model == "wav2vec":
                self.effective_feature_losses = torch.zeros(len(params.wav2vec_loss_layers), device=params.device)
                for idx, layer_num in enumerate(params.wav2vec_loss_layers):
                    self.effective_feature_losses[idx] = feature_losses[layer_num]
                total_feature_loss = torch.sum(self.effective_feature_losses / self.feature_weights)

            # predicted_features_pase = compute_features_pase(pred_wavs)
            # target_features_pase = compute_features_pase(target_wavs)
            # pase_feature_loss = params.mse_cost(predicted_features_pase[0], target_features_pase[0])
            # feature_losses.append(pase_feature_loss)

            if params.combined_loss:
                r1, r2 = self.weights_ratio.split(":")
                loss = float(r1) * basic_loss + float(r2) * total_feature_loss.to(params.device)  # + \
                       # 0.005 * pase_feature_loss.to(params.device)
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
                params.Sample_rate
            )

            ssnrs = compute_ssnr(
                pred_wavs.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
                params.Sample_rate
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
                csigs, cbaks, covls = compute_composite(
                    pred_wavs.cpu().numpy(),
                    target_wavs.cpu().numpy(),
                    np.array([length]),
                    params.Sample_rate
                )
                stats["csigs"] = csigs
                stats["cbaks"] = cbaks
                stats["covls"] = covls

            stats["stoi"] = -stoi_loss(pred_wavs, target_wavs, torch.Tensor([length])).unsqueeze(0)

            if stage == "test":
                enhance_path = os.path.join(params.enhanced_folder, name + ".wav")
                sf.write(enhance_path, pred_wavs.cpu().numpy().squeeze(), params.Sample_rate)
                noisy_path = os.path.join(params.enhanced_folder, name + "_noisy.wav")
                sf.write(noisy_path, input_wavs.cpu().numpy().squeeze(), params.Sample_rate)
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

        loss, stats = self.compute_objectives(pred_wavs, [inputs, targets], stage=stage)
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

batch = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[batch],
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
