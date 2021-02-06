import torch
import torch.nn as nn

from waveunet_utils import crop, Resample1d, ConvLayer


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size


class Waveunet(nn.Module):
    def __init__(self, num_levels, features, kernel_size,
                 target_output_size, conv_type, res,
                 feature_growth="double", depth=1, strides=2):
        super(Waveunet, self).__init__()

        self.num_levels = num_levels
        self.strides = strides
        self.kernel_size = kernel_size
        self.depth = depth

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        module = nn.Module()

        module.downsampling_blocks = nn.ModuleList()
        module.upsampling_blocks = nn.ModuleList()

        num_channels = [features * i for i in range(1, num_levels + 1)] if feature_growth == "add" else \
                       [features * 2 ** i for i in range(0, num_levels)]
        for i in range(self.num_levels - 1):
            in_ch = 1 if i == 0 else num_channels[i]

            module.downsampling_blocks.append(
                DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

        for i in range(0, self.num_levels - 1):
            module.upsampling_blocks.append(
                UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

        module.bottlenecks = nn.ModuleList(
            [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

        # Output conv
        module.output_conv = nn.Conv1d(num_channels[0], 1, 1)
        self.module = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        try:
            curr_size = bottleneck
            for idx, block in enumerate(self.module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(self.module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(self.module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward(self, x):
        curr_input_size = x.shape[-1]
        # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size
        assert (curr_input_size == self.input_size)

        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in self.module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(self.module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = self.module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

