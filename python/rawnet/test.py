from model_RawNet2_original_code import RawNet
from parser import get_args
import torch

import torch.nn.functional as F

args = get_args()
args.model['nb_classes'] = 6112
rawnet = RawNet(args.model)
rawnet.load_state_dict(torch.load('./rawnet2_best_weights.pt'))
rawnet.to(torch.device('cuda'))
rawnet.eval()

def compute_features_rawnet(x):
    nb_samp = x.shape[0]
    len_seq = x.shape[1]
    x = rawnet.ln(x)
    x = x.view(nb_samp, 1, len_seq)
    x = F.max_pool1d(torch.abs(rawnet.first_conv(x)), 3)
    x = rawnet.first_bn(x)
    x = rawnet.lrelu_keras(x)

    x0 = rawnet.block0(x)
    y0 = rawnet.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
    y0 = rawnet.fc_attention0(y0)
    y0 = rawnet.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
    x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

    x1 = rawnet.block1(x)
    y1 = rawnet.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
    y1 = rawnet.fc_attention1(y1)
    y1 = rawnet.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
    x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

    return x

wav_input_16khz = torch.randn(1, 59049, device=torch.device('cuda'))
result = compute_features_rawnet(wav_input_16khz)
print(result.shape)