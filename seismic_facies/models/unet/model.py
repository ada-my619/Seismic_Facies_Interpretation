import torch
import torch.nn as nn
import torch.nn.functional as F

from seismic_facies.models.helper import pad_to_match

class ConvBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super().__init__()
    self.c = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1)
    self.bn = nn.GroupNorm(8, output_channel)
    self.activation = nn.SiLU()

  def forward(self, x):
    x = self.c(x)
    x = self.bn(x)
    x = self.activation(x)
    return x

class DecBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super().__init__()
    self.up = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2, padding=0)
    self.conv_block1 = ConvBlock(output_channel * 2, output_channel) # 2 times for handling skip connection, double the channel size since we concat on the channel dim
    self.conv_block2 = ConvBlock(output_channel, output_channel)

  def forward(self, x, s):
    h = self.up(x)

    if h.shape[-2:] != s.shape[-2:]:
        h = pad_to_match(h, s)
    h = torch.cat([h, s], dim=1)
    h = self.conv_block1(h)
    h = self.conv_block2(h)
    return h

class EncBlock(nn.Module):
  def __init__(self, input_channel, output_channel):
    super().__init__()
    self.conv_block1 = ConvBlock(input_channel, output_channel)
    self.conv_block2 = ConvBlock(output_channel, output_channel)
    self.pool = nn.MaxPool2d((2, 2))

  def forward(self, x):
    h = self.conv_block1(x)
    h = self.conv_block2(h)
    p = self.pool(h)
    return h, p


class UNet(nn.Module):
    def __init__(self, in_channels = 1, n_classes = 6):
      super().__init__()
      self.e1 = EncBlock(in_channels, 8)
      self.e2 = EncBlock(8, 16)
      self.e3 = EncBlock(16, 32)
      self.e4 = EncBlock(32, 64)
      self.e5 = EncBlock(64, 128)
      self.b1 = nn.Sequential(
          ConvBlock(128, 256),
          nn.Dropout2d(p=0.2)
          )
      self.d1 = DecBlock(256, 128)
      self.d2 = DecBlock(128, 64)
      self.d3 = DecBlock(64, 32)
      self.d4 = DecBlock(32, 16)
      self.d5 = DecBlock(16, 8)

      self.output = nn.Conv2d(8, n_classes, kernel_size=1, padding=0) # we keep the size, just reduce the channels

    def forward(self, x):
      B, C, H, W = x.shape
      #print(B, C, H, W)
      pad_h = (32 - H % 32) % 32
      pad_w = (32 - W % 32) % 32

      x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)

      s1, x = self.e1(x)
      s2, x = self.e2(x)
      s3, x = self.e3(x)
      s4, x = self.e4(x)
      s5, x = self.e5(x)
      x = self.b1(x)
      x = self.d1(x, s5)
      x = self.d2(x, s4)
      x = self.d3(x, s3)
      x = self.d4(x, s2)
      x = self.d5(x, s1)
      output = self.output(x)
      output = output[:, :, :H, :W]
      # print(output.shape)
      return output
    

class SelfAttention2D(nn.Module):
    """
    Simple self-attention over spatial positions.
    x: (B, C, H, W) -> (B, C, H, W)
    """
    def __init__(self, channels, num_heads=4, dropout=0.0, gn_groups=8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.GroupNorm(gn_groups, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        res = x

        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)  # (B, N, C)

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.ff(x)

        x = x.transpose(1, 2).view(b, c, h, w)
        return res + x
    
class UNetAttn(nn.Module):
    def __init__(self, in_channels=1, n_classes=6):
        super().__init__()
        self.e1 = EncBlock(in_channels, 8)
        self.e2 = EncBlock(8, 16)
        self.e3 = EncBlock(16, 32)
        self.e4 = EncBlock(32, 64)
        self.e5 = EncBlock(64, 128)

        self.b1 = nn.Sequential(
            ConvBlock(128, 256),
            nn.Dropout2d(p=0.2)
        )

        self.d1 = DecBlock(256, 128)
        self.d2 = DecBlock(128, 64)
        self.d3 = DecBlock(64, 32)
        self.d4 = DecBlock(32, 16)
        self.d5 = DecBlock(16, 8)

        self.output = nn.Conv2d(8, n_classes, kernel_size=1)

        # attention modules
        # e5 output has 128 channels; d1 output has 128 channels; bottleneck has 256 channels.
        self.attn_e5 = SelfAttention2D(channels=128, num_heads=8)
        self.attn_bot = SelfAttention2D(channels=256, num_heads=8)
        self.attn_d1 = SelfAttention2D(channels=128, num_heads=8)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        x = F.pad(x, (0, pad_w, 0, pad_h))

        s1, x = self.e1(x)
        s2, x = self.e2(x)
        s3, x = self.e3(x)
        s4, x = self.e4(x)

        s5, x = self.e5(x)

        # attention on e5 skip
        s5 = self.attn_e5(s5)

        x = self.b1(x)

        # attention at bottleneck
        x = self.attn_bot(x)

        x = self.d1(x, s5)

        # attention after d1
        #x = self.attn_d1(x)

        x = self.d2(x, s4)
        x = self.d3(x, s3)
        x = self.d4(x, s2)
        x = self.d5(x, s1)

        output = self.output(x)
        output = output[:, :, :H, :W]
        return output