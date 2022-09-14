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
    def __init__(self, ):
        super(EncoderDecoder, self).__init__()
        self.encoder_1 = EncoderUnit(8, 128, 128, 128)
        self.encoder_2 = EncoderUnit(128, 128, 128, 128)
        self.encoder_3 = EncoderUnit(128, 128, 128, 128)
        self.encoder_4 = EncoderUnit(128, 128, 128, 128)
        self.encoder_5 = EncoderUnit(128, 128, 128, 128)
        self.decoder_5 = DecoderUnit(128, 128, 128, 128)
        self.decoder_4 = DecoderUnit(144, 128, 128, 128)
        self.decoder_3 = DecoderUnit(144, 128, 128, 128)
        self.decoder_2 = DecoderUnit(144, 128, 128, 128)
        self.decoder_1 = DecoderUnit(144, 128, 128, 128)
        self.skipconnection_1 = SkipConnection(8, 16)
        self.skipconnection_2 = SkipConnection(128, 16)
        self.skipconnection_3 = SkipConnection(128, 16)
        self.skipconnection_4 = SkipConnection(128, 16)
        self.skipconnection_5 = SkipConnection(128, 16)
        self.final_conv = nn.Conv2d(144, 1, (1, 1), padding=0)
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
  def __init__(self):
    super(FCN, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(200, 1000),
        nn.ReLU(),
        nn.Linear(1000,961),
        nn.Softmax()
    )
  def forward(self, input):

    return self.net(input)