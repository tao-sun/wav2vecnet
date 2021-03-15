#!/usr/bin/python

"""
Recipe to train CONV-TASNET model on the WSJ0 dataset

Author:
    * Cem Subakan 2020
"""

import os, sys
import speechbrain as sb
from speechbrain.utils.train_logger import summarize_average
import torch
from torch.utils.data import DataLoader
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

import torch.nn.functional as F
from utils import worker_init_fn
from prepare_data.hdf5_prepare import create_hdf5

import numpy as np

from evaluate.util import compute_pesq, compute_composite, compute_ssnr, sisdr, snr

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from prepare_data.timit_prepare import prepare_timit
from prepare_data.voicebank_prepare import prepare_voicebank

import soundfile as sf

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


class CTN_Brain(sb.core.Brain):
    def compute_forward(self, batch, stage="train", init_params=False):
        # if hasattr(params, "env_corrupt"):
        #     if stage == "train":
        #         wav_lens = torch.tensor(
        #             [mixture.shape[-1]] * mixture.shape[0]
        #         ).to(params.device)
        #         mixture = params.augmentation(mixture, wav_lens, init_params)

        inputs, targets = batch
        mixture = torch.squeeze(inputs[1].to(params.device), 1)
        clean = torch.squeeze(targets[1].to(params.device), 1)
        noise = mixture - clean
        targets = torch.cat([clean.unsqueeze(-1), noise.unsqueeze(-1)], dim=-1)
        # if stage == "train" and params.limit_training_signal_len:
        #     with torch.no_grad():
        #         mixture, targets = self.cut_signals(mixture, targets)

        mixture_w = params.Encoder(mixture, init_params)
        est_mask = params.MaskNet(mixture_w, init_params)
        est_source = params.Decoder(mixture_w, est_mask, init_params)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        if T_origin > T_conv:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_conv))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets, stage="train"):
        if params.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
        else:
            raise ValueError("Not Correct Loss Function Type")

        return loss

    def fit_batch(self, batch):


        # train_onthefly option enables data augmentation, by creating random mixtures within the batch
        # if params.train_onthefly:
        #     bs = batch[0][1].shape[0]
        #     perm = torch.randperm(bs)
        #
        #     T = 24000
        #     Tmax = max((batch[0][1].shape[-1] - T) // 10, 1)
        #     Ts = torch.randint(0, Tmax, (1,))
        #     source1 = batch[1][1][perm, Ts : Ts + T].to(device)
        #     source2 = batch[2][1][:, Ts : Ts + T].to(device)
        #
        #     ws = torch.ones(2).to(device)
        #     ws = ws / ws.sum()
        #
        #     inputs = ws[0] * source1 + ws[1] * source2
        #     targets = torch.cat(
        #         [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
        #     )
        # else:
        # inputs, targets = batch
        # inputs = batch[0][1].to(device)
        # targets = torch.cat(
        #     [batch[1][1].unsqueeze(-1), (batch[0][1] - batch[1][1]).unsqueeze(-1)], dim=-1
        # ).to(device)

        predictions, targets = self.compute_forward(batch)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        inputs, targets = batch
        _, input_wavs, _ = inputs
        names, target_wavs, length = targets

        batch = [(None, batch[0][1].squeeze(0), None), (None, batch[1][1].squeeze(0), None)]
        predictions, targets = self.compute_forward(batch, stage=stage)
        # predictions = torch.reshape(predictions, (1, 1, predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))
        # targets = torch.reshape(targets, (1, 1, targets.shape[0] * targets.shape[1] * targets.shape[2]))
        loss = self.compute_objectives(predictions, targets, stage)

        stats = {}
        if stage != "train":
            input_wavs = torch.reshape(input_wavs, (1, input_wavs.shape[1] * input_wavs.shape[2] * input_wavs.shape[3]))
            pred_wavs = torch.reshape(predictions[:, :, 1], (1, predictions.shape[0] * predictions.shape[1]))
            target_wavs = torch.reshape(targets[:, :, 0], (1, targets.shape[0] * targets.shape[1]))

            pesq_scores = compute_pesq(
                pred_wavs.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
                params.sample_rate
            )

            ssnrs = compute_ssnr(
                pred_wavs.cpu().numpy(),
                target_wavs.cpu().numpy(),
                np.array([length]),
                params.sample_rate
            )

            stats["basic_loss"] = loss

            # stats["feature_loss"] = total_feature_loss
            # for i, feature_loss in enumerate(feature_losses):
            #     stats["fl[{idx}]".format(idx=i)] = feature_loss

            stats['snr'] = [snr(pred_wavs, target_wavs, False)]
            stats["ssnrs"] = ssnrs
            stats['si_sdr'] = [sisdr(pred_wavs, target_wavs)]
            stats["pesq"] = pesq_scores
            if stage == "test":
                csigs, cbaks, covls = compute_composite(
                    pred_wavs.cpu().numpy(),
                    target_wavs.cpu().numpy(),
                    np.array([length]),
                    params.sample_rate
                )
                stats["csigs"] = csigs
                stats["cbaks"] = cbaks
                stats["covls"] = covls

            stats["stoi"] = -stoi_loss(pred_wavs, target_wavs, torch.Tensor([length])).unsqueeze(0)

            if stage == "test":
                enhance_path = os.path.join(params.enhanced_folder, names[0] + ".wav")
                sf.write(enhance_path, pred_wavs.cpu().numpy().squeeze(), params.sample_rate)
                noisy_path = os.path.join(params.enhanced_folder, names[0] + "_noisy.wav")
                sf.write(noisy_path, input_wavs.cpu().numpy().squeeze(), params.sample_rate)
                clean_path = os.path.join(params.enhanced_folder, names[0] + "_clean.wav")
                sf.write(clean_path, target_wavs.cpu().numpy().squeeze(), params.sample_rate)

        stats["loss"] = loss.detach()
        return stats

    def on_epoch_end(self, epoch, train_stats, valid_stats):
        av_loss = summarize_average(valid_stats["loss"])
        if params.use_tensorboard:
            params.train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)

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
            test_stats = self.evaluate(test_loader)
            params.train_logger.log_stats(
                stats_meta={"Epoch loaded": params.epoch_counter.current},
                test_stats=test_stats,
            )

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - params.training_signal_len),
            (1,),
        ).item()
        targets = targets[
                  :, randstart: randstart + params.training_signal_len, :
                  ]
        mixture = mixture[
                  :, randstart: randstart + params.training_signal_len
                  ]
        return mixture, targets


if not os.path.exists(params.hdf5_train):
    create_hdf5(params.csv_train, params.hdf5_train, params.sample_rate)

train_loader = DataLoader(params.train_dataset,
                          batch_size=params.batch_size,
                          shuffle=True,
                          num_workers=1,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False)
valid_loader = DataLoader(params.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False)
test_loader = DataLoader(params.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=1,
                          worker_init_fn=worker_init_fn,
                          pin_memory=False)
first_batch = next(iter(train_loader))

ctn = CTN_Brain(
    modules=[
        params.Encoder.to(params.device),
        params.MaskNet.to(params.device),
        params.Decoder.to(params.device),
    ],
    optimizer=params.optimizer,
    first_inputs=[first_batch],
)

params.checkpointer.recover_if_possible(lambda c: -c.meta["pesq_score"])

ctn.fit(
    range(params.N_epochs),
    train_set=train_loader,
    valid_set=valid_loader,
    progressbar=params.progressbar
)

test_stats = ctn.evaluate(test_loader)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats
)
