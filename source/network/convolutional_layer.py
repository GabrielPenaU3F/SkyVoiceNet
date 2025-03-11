import torch.nn as nn


class ConvolutionalLayer(nn.Module):

    def __init__(self, out_channels):
        super(ConvolutionalLayer, self).__init__()
        self.conv = None
        # self.norm = nn.BatchNorm2d(out_channels)
        if out_channels == 1:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class ConvolutionalEncoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
        super().__init__(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)


class ConvolutionalDecoderLayer(ConvolutionalLayer):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0),
                 output_padding=(0, 0)):
        super().__init__(out_channels)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)
