import torch
import torch.nn.functional as F

from models import Cnn14, Wavegram_Cnn14
import config
from parser import get_args

args = get_args()
sample_rate = args.sample_rate
window_size = args.window_size
hop_size = args.hop_size
mel_bins = args.mel_bins
fmin = args.fmin
fmax = args.fmax

device = torch.device('cuda')

classes_num = config.classes_num
print("classes_num: " + str(classes_num))
labels = config.labels

Model = Cnn14
model = Model(sample_rate=sample_rate, window_size=window_size,
              hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
              classes_num=classes_num)

checkpoint = torch.load("/home/tao/Projects/wav2vecnet/data/pretrained/Cnn14_16k.pth", map_location=device)
model.load_state_dict(checkpoint['model'])

model.to(device)
model.eval()

def compute_features_audioset_tagging_cnn(x):
    x = model.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
    x = model.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

    x = x.transpose(1, 3)
    x = model.bn0(x)
    x = x.transpose(1, 3)

    if model.training:
        x = model.spec_augmenter(x)

    x = model.conv_block1(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=model.training)
    x = model.conv_block2(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=model.training)
    x = model.conv_block3(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=model.training)
    x = model.conv_block4(x, pool_size=(2, 2), pool_type='avg')
    x = F.dropout(x, p=0.2, training=model.training)

    return x


wav_input_16khz = torch.randn(1, 10000, device=device)
result = compute_features_audioset_tagging_cnn(wav_input_16khz)
print(result.shape)

# result = model(wav_input_16khz, None)
# print(result.keys())
