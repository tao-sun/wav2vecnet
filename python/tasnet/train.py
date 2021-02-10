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
from speechbrain.utils.checkpoints import ckpt_recency
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

import torch.nn.functional as F

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from prepare_data.timit_prepare import prepare_timit
from prepare_data.voicebank_prepare import prepare_voicebank

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
    def compute_forward(self, inputs, stage="train", init_params=False):

        # if hasattr(params, "env_corrupt"):
        #     if stage == "train":
        #         wav_lens = torch.tensor(
        #             [mixture.shape[-1]] * mixture.shape[0]
        #         ).to(params.device)
        #         mixture = params.augmentation(mixture, wav_lens, init_params)

        mixture = inputs.to(params.device)
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

        return est_source

    def compute_objectives(self, predictions, targets):
        if params.loss_fn == "sisnr":
            loss = get_si_snr_with_pitwrapper(targets, predictions)
            return loss
        else:
            raise ValueError("Not Correct Loss Function Type")

    def fit_batch(self, batch):
        device = params.device
        # train_onthefly option enables data augmentation, by creating random mixtures within the batch
        if params.train_onthefly:
            bs = batch[0][1].shape[0]
            perm = torch.randperm(bs)

            T = 24000
            Tmax = max((batch[0][1].shape[-1] - T) // 10, 1)
            Ts = torch.randint(0, Tmax, (1,))
            source1 = batch[1][1][perm, Ts : Ts + T].to(device)
            source2 = batch[2][1][:, Ts : Ts + T].to(device)

            ws = torch.ones(2).to(device)
            ws = ws / ws.sum()

            inputs = ws[0] * source1 + ws[1] * source2
            targets = torch.cat(
                [source1.unsqueeze(1), source2.unsqueeze(1)], dim=1
            )
        else:
            inputs = batch[0][1].to(device)
            targets = torch.cat(
                [batch[1][1].unsqueeze(-1), (batch[0][1] - batch[1][1]).unsqueeze(-1)], dim=-1
            ).to(device)

        predictions = self.compute_forward(inputs)
        loss = self.compute_objectives(predictions, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach()}

    def evaluate_batch(self, batch, stage="test"):
        device = params.device

        inputs = batch[0][1].to(device)
        targets = torch.cat(
            [batch[1][1].unsqueeze(-1), batch[2][1].unsqueeze(-1)], dim=-1
        ).to(device)

        predictions = self.compute_forward(inputs, stage="test")
        loss = self.compute_objectives(predictions, targets)
        return {"loss": loss.detach()}

    def on_epoch_end(self, epoch, train_stats, valid_stats):

        av_loss = summarize_average(valid_stats["loss"])
        if params.use_tensorboard:
            params.train_logger.log_stats({"Epoch": epoch}, train_stats, valid_stats)
        print("Completed epoch %d" % epoch)
        print("Train SI-SNR: %.3f" % -summarize_average(train_stats["loss"]))
        print("Valid SI-SNR: %.3f" % -summarize_average(valid_stats["loss"]))

        params.checkpointer.save_and_keep_only(
            meta={"av_loss": av_loss},
            importance_keys=[ckpt_recency, lambda c: -c.meta["av_loss"]],
        )


train_loader = params.train_loader()
valid_loader = params.valid_loader()
test_loader = params.test_loader()

first_x, first_y = next(iter(train_loader))

ctn = CTN_Brain(
    modules=[
        params.Encoder.to(params.device),
        params.MaskNet.to(params.device),
        params.Decoder.to(params.device),
    ],
    optimizer=params.optimizer,
    first_inputs=[first_x[1]],
)

params.checkpointer.recover_if_possible(lambda c: -c.meta["av_loss"])

ctn.fit(
    range(params.N_epochs),
    train_set=train_loader,
    valid_set=valid_loader,
    progressbar=params.progressbar
)

test_stats = ctn.evaluate(test_loader)
print("Test SI-SNR: %.3f" % -summarize_average(test_stats["loss"]))
