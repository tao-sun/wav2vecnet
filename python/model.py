import torch
import torch.nn as NN
import torch.nn.functional as F

from fairseq.models.wav2vec import Wav2VecModel


class Wav2vecNet(NN.Module):
    def __init__(self, filter_down=31, filter_up=5):
        super(Wav2vecNet, self).__init__()

        # Model Architecture
        # DOWN SAMPLING(CONVLOLUTION)
        self.conv1 = NN.Conv1d(1, 16, kernel_size=filter_down, padding=int((filter_down - 1) / 2))
        self.conv2 = NN.Conv1d(16, 32, kernel_size=filter_down, padding=int((filter_down - 1) / 2))
        self.conv3 = NN.Conv1d(32, 64, kernel_size=filter_down, padding=int((filter_down - 1) / 2))
        self.conv4 = NN.Conv1d(64, 128, kernel_size=filter_down, padding=int((filter_down - 1) / 2))
        self.conv5 = NN.Conv1d(128, 256, kernel_size=filter_down, padding=int((filter_down - 1) / 2))
        self.conv6 = NN.Conv1d(256, 512, kernel_size=filter_down, padding=int((filter_down - 1) / 2))

        # UP SAMPLING(TRANSPOSED CONVLOLUTION)
        self.conv_up0 = NN.Conv1d(512+256, 256, kernel_size=filter_up, padding=int((filter_up - 1) / 2))
        self.conv_up1 = NN.Conv1d(256+128, 128, kernel_size=filter_up, padding=int((filter_up - 1) / 2))
        self.conv_up2 = NN.Conv1d(128+64, 64, kernel_size=filter_up, padding=int((filter_up - 1) / 2))
        self.conv_up3 = NN.Conv1d(64+32, 32, kernel_size=filter_up, padding=int((filter_up - 1) / 2))
        self.conv_up4 = NN.Conv1d(32+16, 16, kernel_size=filter_up, padding=int((filter_up - 1) / 2))
        self.conv_up5 = NN.Conv1d(16+1, 1, kernel_size=filter_up, padding=int((filter_up - 1) / 2))

        cp = torch.load('./data/pretrained/wav2vec_large.pt')
        self.wav2vec_model = Wav2VecModel.build_model(cp['args'], task=None)

    def forward(self, input):
        batch_size, num_frames, _ = input.size()
        input = input.view(batch_size * num_frames, 1, self.cnn_input_size)

        ######### Down sampling #############
        c1 = F.leaky_relu(self.conv1(input))
        c1_pool = c1[:,:,::2]

        c2 = F.leaky_relu(self.conv2(c1_pool))
        c2_pool = c2[:,:,::2]

        c3 = F.leaky_relu(self.conv3(c2_pool))
        c3_pool = c3[:,:,::2]

        c4 = F.leaky_relu(self.conv4(c3_pool))
        c4_pool = c4[:,:,::2]

        c5 = F.leaky_relu(self.conv5(c4_pool))
        c5_pool = c5[:,:,::2]

        c6 = F.leaky_relu(self.conv6(c5_pool))

        ######### UP sampling ###############
        u0 = F.upsample(c6, scale_factor=2, mode='linear')
        u0_1 = F.leaky_relu(self.conv_up0(torch.cat((u0, c5), dim=1)))

        u1 = F.upsample(u0_1, scale_factor=2, mode='linear')
        u1_1 = F.leaky_relu(self.conv_up1(torch.cat((u1, c4), dim=1)))

        u2 = F.upsample(u1_1, scale_factor=2, mode='linear')
        u2_1 = F.leaky_relu(self.conv_up2(torch.cat((u2, c3), dim=1)))

        u3 = F.upsample(u2_1, scale_factor=2, mode='linear')
        u3_1 = F.leaky_relu(self.conv_up3(torch.cat((u3, c2), dim=1)))

        u4 = F.upsample(u3_1, scale_factor=2, mode='linear')
        u4_1 = F.leaky_relu(self.conv_up4(torch.cat((u4, c1), dim=1)))

        output = F.tanh(self.conv_up5(torch.cat((u4_1, input), dim=1)))
        return output
