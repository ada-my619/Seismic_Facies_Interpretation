import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to_match(src, ref):
    """
    Make src match ref spatially by center-cropping if too big, or symmetric padding if too small.
    src, ref: (B,C,H,W)
    """
    _, _, Hs, Ws = src.shape
    _, _, Hr, Wr = ref.shape

    # center-crop if src is bigger
    if Hs > Hr:
        top = (Hs - Hr) // 2
        src = src[:, :, top:top + Hr, :]
    if Ws > Wr:
        left = (Ws - Wr) // 2
        src = src[:, :, :, left:left + Wr]

    # symmetric pad if src is smaller
    _, _, Hs, Ws = src.shape
    dh, dw = Hr - Hs, Wr - Ws
    if dh > 0 or dw > 0:
        pad_top = dh // 2
        pad_bottom = dh - pad_top
        pad_left = dw // 2
        pad_right = dw - pad_left
        src = F.pad(src, (pad_left, pad_right, pad_top, pad_bottom))
    return src

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