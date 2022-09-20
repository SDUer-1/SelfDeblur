import torch
import torch.nn as nn
from utils import *

# encoder unit
class EncoderUnit(nn.Module):
    def __init__(self, conv_1_input_c, conv_1_output_c, conv_2_input_c, conv_2_output_c, conv_1_ksize=(3, 3),
                 conv_2_ksize=(3, 3), conv_1_padding=1, conv_2_padding=1):
        super(EncoderUnit, self).__init__()
        self.padder_1 = nn.ReflectionPad2d(conv_1_padding)
        self.padder_2 = nn.ReflectionPad2d(conv_2_padding)
        self.conv_1_layer = nn.Conv2d(conv_1_input_c, conv_1_output_c, conv_1_ksize, stride=2)
        self.conv_2_layer = nn.Conv2d(conv_2_input_c, conv_2_output_c, conv_2_ksize)
        self.bn_1 = nn.BatchNorm2d(conv_1_output_c)
        self.bn_2 = nn.BatchNorm2d(conv_2_output_c)
        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.padder_1(x)
        x = self.conv_1_layer(x)
        x = self.bn_1(x)
        x = self.leakyReLU(x)
        x = self.padder_2(x)
        x = self.conv_2_layer(x)
        x = self.bn_2(x)
        x = self.leakyReLU(x)
        return x

# decoder unit
class DecoderUnit(nn.Module):
    def __init__(self, conv_1_input_c, conv_1_output_c, conv_2_input_c, conv_2_output_c, conv_1_ksize=(3, 3),
                 conv_1_padding=1):
        super(DecoderUnit, self).__init__()
        self.padder_1 = nn.ReflectionPad2d(conv_1_padding)
        self.conv_1_layer = nn.Conv2d(conv_1_input_c, conv_1_output_c, conv_1_ksize)
        self.conv_2_layer = nn.Conv2d(conv_2_input_c, conv_2_output_c, (1, 1))
        self.bn_1 = nn.BatchNorm2d(conv_1_input_c)
        self.bn_2 = nn.BatchNorm2d(conv_1_output_c)
        self.bn_3 = nn.BatchNorm2d(conv_2_output_c)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.upsampling = nn.Upsample(scale_factor=2.0, mode='bilinear')

    def forward(self, x):
        x = self.bn_1(x)
        x = self.padder_1(x)
        x = self.conv_1_layer(x)
        x = self.bn_2(x)
        x = self.leakyReLU(x)
        x = self.conv_2_layer(x)
        x = self.bn_2(x)
        x = self.leakyReLU(x)
        x = self.upsampling(x)
        return x

# skip connection
class SkipConnection(nn.Module):
    def __init__(self, conv_input_c, conv_output_c, conv_ksize=(1, 1)):
        super(SkipConnection, self).__init__()
        self.conv_layer = nn.Conv2d(conv_input_c, conv_output_c, conv_ksize)
        self.bn_1 = nn.BatchNorm2d(conv_output_c)
        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_1(x)
        x = self.leakyReLU(x)
        return x

# Gx
class EncoderDecoder(nn.Module):
    def __init__(self, input_channels=8, encoder_channels=[128,128,128,128,128], decoder_channels=[128,128,128,128,128], skip_channels=[16,16,16,16,16]):
        '''

        :param input_channels: channels of input
        :param encoder_channels: list of channels in encoders
        :param decoder_channels: list of channels in decoders
        :param skip_channels: list of channels in skip connections
        '''
        super(EncoderDecoder, self).__init__()
        self.encoder_1 = EncoderUnit(input_channels, encoder_channels[0], encoder_channels[0], encoder_channels[1])
        self.encoder_2 = EncoderUnit(encoder_channels[1], encoder_channels[1], encoder_channels[1], encoder_channels[2])
        self.encoder_3 = EncoderUnit(encoder_channels[2], encoder_channels[2], encoder_channels[2], encoder_channels[3])
        self.encoder_4 = EncoderUnit(encoder_channels[3], encoder_channels[3], encoder_channels[3], encoder_channels[4])
        self.encoder_5 = EncoderUnit(encoder_channels[4], encoder_channels[4], encoder_channels[4], decoder_channels[0])
        self.decoder_5 = DecoderUnit(decoder_channels[0], decoder_channels[0], decoder_channels[0], decoder_channels[1])
        self.decoder_4 = DecoderUnit(decoder_channels[1] + skip_channels[0], decoder_channels[1], decoder_channels[1], decoder_channels[2])
        self.decoder_3 = DecoderUnit(decoder_channels[2] + skip_channels[1], decoder_channels[2], decoder_channels[2], decoder_channels[3])
        self.decoder_2 = DecoderUnit(decoder_channels[3] + skip_channels[2], decoder_channels[3], decoder_channels[3], decoder_channels[4])
        self.decoder_1 = DecoderUnit(decoder_channels[4] + skip_channels[3], decoder_channels[4], decoder_channels[4], decoder_channels[4])
        self.skipconnection_1 = SkipConnection(input_channels, skip_channels[0])
        self.skipconnection_2 = SkipConnection(encoder_channels[1], skip_channels[1])
        self.skipconnection_3 = SkipConnection(encoder_channels[2], skip_channels[2])
        self.skipconnection_4 = SkipConnection(encoder_channels[3], skip_channels[3])
        self.skipconnection_5 = SkipConnection(encoder_channels[4], skip_channels[4])
        self.final_conv = nn.Conv2d(decoder_channels[4] + skip_channels[4], 1, (1, 1), padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        skip_x_1 = self.skipconnection_1(x)
        x = self.encoder_1(x)
        skip_x_2 = self.skipconnection_2(x)
        x = self.encoder_2(x)
        skip_x_3 = self.skipconnection_3(x)
        x = self.encoder_3(x)
        skip_x_4 = self.skipconnection_4(x)
        x = self.encoder_4(x)
        skip_x_5 = self.skipconnection_5(x)
        x = self.encoder_5(x)
        # Decoder
        x = self.decoder_5(x)
        x, skip_x_5 = change_dimension(x, skip_x_5)
        x = torch.cat((x, skip_x_5), dim=1)
        x = self.decoder_4(x)
        x, skip_x_4 = change_dimension(x, skip_x_4)
        x = torch.cat((x, skip_x_4), dim=1)
        x = self.decoder_3(x)
        x, skip_x_3 = change_dimension(x, skip_x_3)
        x = torch.cat((x, skip_x_3), dim=1)
        x = self.decoder_2(x)
        x, skip_x_2 = change_dimension(x, skip_x_2)
        x = torch.cat((x, skip_x_2), dim=1)
        x = self.decoder_1(x)
        x, skip_x_1 = change_dimension(x, skip_x_1)
        x = torch.cat((x, skip_x_1), dim=1)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

# Gk
class FCN(nn.Module):
  def __init__(self, m_k, n_k):
    '''

    :param m_k: shape[0] of kernel
    :param n_k: shape[1] of kernel
    '''
    super(FCN, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(200, 1000),
        nn.ReLU(),
        nn.Linear(1000,m_k * n_k),
        nn.Softmax()
    )
  def forward(self, input):

    return self.net(input)