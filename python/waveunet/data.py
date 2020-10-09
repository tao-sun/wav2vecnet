import h5py
import csv
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import glob

# {'output_start_frame': 312, 'output_end_frame': 32321, 'output_frames': 32009, 'input_frames': 32633}
class SeparationDataset(Dataset):
    def __init__(self, hdf_file, sr, channels, random_hops):
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
        self.shapes = None
        self.hdf_dataset = None

    def set_shapes(self, shapes):
        self.shapes = shapes

    def __getitem__(self, index):
        if self.hdf_dataset is None:
            self.hdf_dataset = h5py.File(self.hdf_file, 'r')

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        name = self.hdf_dataset[str(index)].attrs["ID"]
        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        clean_length = self.hdf_dataset[str(audio_idx)].attrs["clean_length"]

        # Determine position where to start clean
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(clean_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["noisy"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        clean_audio = self.hdf_dataset[str(audio_idx)]["clean"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            clean_audio = np.pad(clean_audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

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
                lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

            self.start_pos = SortedList(np.cumsum(lengths))
            self.length = self.start_pos[-1]
        return self.length


class EvaluationDataset(Dataset):
    def __init__(self, hdf_file, sr, channels):

        super(EvaluationDataset, self).__init__()

        self.sr = sr
        self.hdf_file = hdf_file
        self.channels = channels
        self.shapes = None
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

        output_shift = self.shapes["output_frames"]

        pad_back = audio.shape[1] % output_shift
        pad_back = 0 if pad_back == 0 else output_shift - pad_back
        if pad_back > 0:
            audio = np.pad(audio, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)
            clean = np.pad(clean, [(0, 0), (0, pad_back)], mode="constant", constant_values=0.0)

        target_outputs = audio.shape[1]

        # Pad mixture across time at beginning and end so that neural network can make prediction at the beginning and end of signal
        pad_front_context = self.shapes["output_start_frame"]
        pad_back_context = self.shapes["input_frames"] - self.shapes["output_end_frame"]
        audio = np.pad(audio, [(0, 0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)
        clean = np.pad(clean, [(0, 0), (pad_front_context, pad_back_context)], mode="constant", constant_values=0.0)

        # Iterate over mixture magnitudes, fetch network prediction
        examples = [audio[:, target_start_pos:target_start_pos + self.shapes["input_frames"]]
                    for target_start_pos in range(0, target_outputs, self.shapes["output_frames"])]
        # targets = [clean[:, target_start_pos:target_start_pos + self.shapes["input_frames"]]
        #            for target_start_pos in range(0, target_outputs, self.shapes["output_frames"])]

        return [name, np.array(examples), audio_length], [name, clean, clean_length]

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
