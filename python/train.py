#!/usr/bin/python
import math
import os
import sys

import torch
import torchaudio
from joblib import Parallel, delayed
from pesq import pesq
from torch import FloatTensor, LongTensor

import speechbrain as sb
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.train_logger import summarize_average

import soundfile as sf

# This hack needed to import data preparation script from ..
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from timit_prepare import prepare_timit  # noqa E402

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


def pad_input(input, frame_length):
    """
    First, pad each item to the longest item in the batch
    Second, pad each item to a multiple of frame_length

    :param input:
    :param frame_length:
    :return:
    """
    batch_lengths = LongTensor(list(map(len, input)))
    frame_num = int(math.ceil(batch_lengths.max().item() / frame_length))
    batch_tensor = torch.zeros((len(input), frame_num * frame_length)).float().cuda()

    for idx, (seq, seq_len) in enumerate(zip(input, batch_lengths)):
        t_seq = FloatTensor(seq)
        batch_tensor[idx, :seq_len] = t_seq

    batch_tensor = batch_tensor.view(len(input) * frame_num, frame_length).cuda()
    return batch_tensor, batch_lengths


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


if params.wav2vec_version == 1.0:
    cp = torch.load('./data/pretrained/wav2vec_large.pt')
    wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec.load_state_dict(cp['model'])
    wav2vec.to(torch.device(params.device))
elif params.wav2vec_version == 2.0:
    cp = torch.load('./data/pretrained/libri960_big.pt')
    wav2vec2 = Wav2Vec2Model.build_model(cp['args'])
    wav2vec2.load_state_dict(cp['model'])
    wav2vec2.to(torch.device(params.device))


def compute_features(x):
    if params.wav2vec_version == 1.0:
        return compute_features1(x)
    elif params.wav2vec_version == 2.0:
        return compute_features2(x)


def compute_features1(x):
    feature_extractor = wav2vec.feature_extractor

    features = []
    x = x.unsqueeze(1)
    for i, conv in enumerate(feature_extractor.conv_layers):
        residual = x
        x = conv(x)
        if feature_extractor.skip_connections and x.size(1) == residual.size(1):
            tsz = x.size(2)
            r_tsz = residual.size(2)
            residual = residual[..., :: r_tsz // tsz][..., :tsz]
            x = (x + residual) * feature_extractor.residual_scale
        features.append(x)

    return features


def compute_features2(x):
    feature_extractor = wav2vec2.feature_extractor

    features = []
    x = x.unsqueeze(1)
    for i, conv in enumerate(feature_extractor.conv_layers):
        x = conv(x)
        features.append(x)

    return features


class SEBrain(sb.core.Brain):
    def compute_forward(self, x, stage="train", init_params=False):
        ids, wavs, lens = x
        wavs, lens = truncate(wavs, lens, params.max_length)
        wavs = torch.unsqueeze(wavs, -1)
        wavs, lens = wavs.to(params.device), lens.to(params.device)
        out = params.model(wavs, init_params=init_params)[:, :, 0]
        return out

    def compute_objectives(self, predictions, targets, stage="train"):
        ids, target_wavs, lens = targets
        target_wavs, lens = truncate(target_wavs, lens, params.max_length)
        target_wavs = target_wavs.to(params.device)
        lens = lens.to(params.device)
        mse_loss = params.compute_cost(predictions, target_wavs, lens)

        predicted_features = compute_features(predictions)
        target_features = compute_features(target_wavs)
        feature_loss = 0
        for i in range(len(predicted_features)):
            feature_loss += params.compute_cost(predicted_features[i], target_features[i])

        loss = feature_loss

        stats = {}
        if stage != "train":
            lens = lens * target_wavs.shape[1]
            pesq_scores = multiprocess_evaluation(
                predictions.cpu().numpy(),
                target_wavs.cpu().numpy(),
                lens.cpu().numpy(),
            )
            stats["pesq"] = pesq_scores
            stats["stoi"] = -stoi_loss(predictions, target_wavs, lens)

            if stage == "test":
                # Write wavs to file
                for name, pred_wav, length in zip(ids, predictions, lens):
                    name += ".wav"
                    enhance_path = os.path.join(params.enhanced_folder, name)
                    # torchaudio.save(
                    #     enhance_path, pred_wav[: int(length)].to("cpu"), 16000
                    # )
                    sf.write(enhance_path, pred_wav[: int(length)].to("cpu"), 16000)

        return loss, stats

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


# Prepare data
prepare_timit(
    data_folder=params.data_folder,
    save_folder=params.save_folder
)
train_set = params.train_loader()
# valid_set = params.valid_loader()
test_set = params.test_loader()
first_x, first_y = next(iter(train_set))

se_brain = SEBrain(
    modules=[params.model], optimizer=params.optimizer, first_inputs=[first_x],
)

# Load latest checkpoint to resume training
params.checkpointer.recover_if_possible()
se_brain.fit(params.epoch_counter, train_set, test_set)

# Load best checkpoint for evaluation
params.checkpointer.recover_if_possible(max_key="pesq_score")
test_stats = se_brain.evaluate(test_set)
params.train_logger.log_stats(
    stats_meta={"Epoch loaded": params.epoch_counter.current},
    test_stats=test_stats,
)
