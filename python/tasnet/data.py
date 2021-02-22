import h5py
import csv
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob


# {'output_start_frame': 312, 'output_end_frame': 32321, 'output_frames': 32009, 'input_frames': 32633}
class SeparationDataset(Dataset):
    def __init__(self, hdf_file, sr, input_frames, channels, random_hops):
        '''

        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.random_hops = random_hops
        self.sr = sr
        self.hdf_file = hdf_file
        self.channels = channels
        self.input_frames = input_frames
        self.hdf_dataset = None

    def __getitem__(self, index):
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_file, 'r')

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            rel_index = index - self.start_pos[audio_idx - 1]
        else:
            rel_index = 0

        name = self.hdf_dataset[str(audio_idx)].attrs["ID"]
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        clean_length = self.hdf_dataset[str(audio_idx)].attrs["clean_length"]

        # Determine position where to start clean
        if self.random_hops:
            start_pos = np.random.randint(0, max(clean_length - self.input_frames + 1, 1))
        else:
            # Map item index to sample position within song
            start_pos = rel_index * self.input_frames

        # Check back padding
        end_pos = start_pos + self.input_frames
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["noisy"][:, start_pos:end_pos].astype(np.float32)
        if pad_back > 0:
            audio = np.pad(audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)
        clean_audio = self.hdf_dataset[str(audio_idx)]["clean"][:, start_pos:end_pos].astype(np.float32)
        if pad_back > 0:
            clean_audio = np.pad(clean_audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)

        return [name, audio, audio_length], [name, clean_audio, clean_length]

    def __len__(self):
        if self.hdf_dataset is None:
            with h5py.File(self.hdf_file, "r") as f:
                if f.attrs["sr"] != self.sr or \
                        f.attrs["channels"] != self.channels:
                    raise ValueError(
                        "Tried to load existing HDF file, but sampling rate and channel are not as expected. "
                        "Did you load an out-dated HDF file?")

                lengths = [f[str(idx)].attrs["clean_length"] for idx in range(len(f))]
                # Subtract input_size from lengths and divide by hop size to determine number of starting positions
                lengths = [(l // self.input_frames) + 1 for l in lengths]

            self.start_pos = SortedList(np.cumsum(lengths))
            # Length of the dataset if number of the total frames (which
            # equals last element of the cumulated lengths)
            self.length = self.start_pos[-1]
        return self.length


class EvaluationDataset(Dataset):
    def __init__(self, hdf_file, sr, input_frames, channels):
        """
        This dataset will keep all frames of an utterance in the same batch.
        :param hdf_file:
        :param sr:
        :param channels:
        """
        super(EvaluationDataset, self).__init__()

        self.sr = sr
        self.hdf_file = hdf_file
        self.channels = channels
        self.input_frames = input_frames
        self.hdf_dataset = None

    def set_shapes(self, shapes):
        self.shapes = shapes

    def __getitem__(self, index):
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_file, 'r')

        audio_length = self.hdf_dataset[str(index)].attrs["length"]
        clean_length = self.hdf_dataset[str(index)].attrs["clean_length"]
        name = self.hdf_dataset[str(index)].attrs["ID"]

        audio = self.hdf_dataset[str(index)]["noisy"][()]
        clean = self.hdf_dataset[str(index)]["clean"][()]

        output_shift = self.input_frames

        pad_back = audio.shape[1] % output_shift
        pad_back = 0 if pad_back == 0 else output_shift - pad_back
        if pad_back > 0:
            audio = np.pad(audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)
            clean = np.pad(clean, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)

        target_outputs = audio.shape[1]

        # Iterate over mixture magnitudes, fetch network prediction
        examples = [audio[:, target_start_pos:target_start_pos + self.input_frames]
                    for target_start_pos in range(0, target_outputs - self.input_frames + 1, self.input_frames)]
        targets = [clean[:, target_start_pos:target_start_pos + self.input_frames]
                   for target_start_pos in range(0, target_outputs - self.input_frames + 1, self.input_frames)]

        return [name, np.array(examples), audio_length], [name, np.array(targets), clean_length]

    def __len__(self):
        if self.hdf_dataset is None:
            with h5py.File(self.hdf_file, "r") as f:
                if f.attrs["sr"] != self.sr or \
                        f.attrs["channels"] != self.channels:
                    raise ValueError(
                        "Tried to load existing HDF file, but sampling rate and channel are not as expected. "
                        "Did you load an out-dated HDF file?")

                self.length = len(f)
        return self.length
