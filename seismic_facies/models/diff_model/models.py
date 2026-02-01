import torch
import torch.nn as nn

from seismic_facies.models.helper import pad_to_match
from seismic_facies.models.diff_model.utils import SinusoidalPositionEmbeddings
import torch.nn.functional as F

# define a convolutional block with time-embedding
class ConvBlock(nn.Module):
    """time-conditioned convolutional block"""

    def __init__(self, in_c, out_c, embed_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.dense = nn.Linear(embed_dim, out_c)  ## reshapes the time embedding length to the number of channels
        self.bn = nn.GroupNorm(num_groups=8, num_channels=out_c)
        self.act = nn.GELU()

    def forward(self, x, t_embed):
        x = self.conv(x)
        x += self.dense(t_embed)[..., None, None]
        x = self.bn(x)
        x = self.act(x)
        return x

# define an encoder block of the U-Net with time-embedding
class EncBlock(nn.Module):
    """time-conditioned U-Net encoder block"""

    def __init__(self, in_c, out_c, embed_dim):
        super().__init__()
        self.conv_block1 = ConvBlock(in_c, out_c, embed_dim)
        self.conv_block2 = ConvBlock(out_c, out_c, embed_dim)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x, t_embed):
        h = self.conv_block1(x, t_embed)
        h = self.conv_block2(h, t_embed)
        p = self.pool(h)
        #print(h.shape, p.shape)
        return h, p

# define an decoder block of the U-Net with time-embedding
class DecBlock(nn.Module):
    """time-conditioned U-Net decoder block"""

    def __init__(self, in_c, skip_c, out_c, embed_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, padding=0, stride=2)
        self.conv_block1 = ConvBlock(out_c
                                     + skip_c
                                     #+ out_c + out_c
                                     , out_c, embed_dim)
        self.conv_block2 = ConvBlock(out_c, out_c, embed_dim)

    def forward(self, x, s, t_embed):
        h = self.up(x)
        # print(h.shape, s.shape)
        if h.shape[-2:] != s.shape[-2:]:
          h = pad_to_match(h, s)
        h = torch.cat([h, s], axis = 1) # concatenate x with U-Net skip connection from encoder
        #print(h.shape, s.shape)
        h = self.conv_block1(h, t_embed)
        h = self.conv_block2(h, t_embed)
        return h
    
class DiffUnet(nn.Module):
    """DDPM U-Net, https://arxiv.org/abs/2006.11239 and https://arxiv.org/abs/1505.04597"""

    def __init__(self, in_channels, out_channels, cond_channels, embed_dim):
        super().__init__()

        # time positional embedding MLP
        self.embed = nn.Sequential(SinusoidalPositionEmbeddings(embed_dim),
                                   nn.Linear(embed_dim, embed_dim),
                                   nn.GELU(),
                                   nn.Linear(embed_dim, embed_dim))

        # encoder
        self.e1 = EncBlock(in_channels + cond_channels, 32, embed_dim)
        self.e2 = EncBlock(32 + 32, 128, embed_dim)
        self.e3 = EncBlock(64 + 128, 256, embed_dim)
        self.e4 = EncBlock(128 + 256, 512, embed_dim)

        # C
        self.c1 = EncBlock(cond_channels, 32, embed_dim)
        self.c2 = EncBlock(32, 64, embed_dim)
        self.c3 = EncBlock(64, 128, embed_dim)
        self.c4 = EncBlock(128, 256, embed_dim)

        # bottleneck
        self.b1 = ConvBlock(512 + 256, 1024, embed_dim)
        self.b2 = ConvBlock(1024, 1024, embed_dim)

        # decoder
        self.d1 = DecBlock(1024, 768, 512, embed_dim)
        self.d2 = DecBlock(512, 384, 256, embed_dim)
        self.d3 = DecBlock(256, 192, 128, embed_dim)
        self.d4 = DecBlock(128, 64, 64, embed_dim)

        # output layer
        self.output_eps = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x, t, cond):
        B, C, H, W = x.shape
        #print(B, C, H, W)
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16

        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
        cond = F.pad(cond, (0, pad_w, 0, pad_h))
        t_embed = self.embed(t)

        # encoder
        #print(x.shape, cond.shape)
        x = torch.cat([x, cond], dim=1)
        s1, x = self.e1(x, t_embed)
        #print(f"self.e1 = s1 shape: {s1.shape}, x shape: {x.shape}")
        f1, cond = self.c1(cond, t_embed)
        #print(f"self.c1 = f1 shape: {f1.shape}, cond shape: {cond.shape}")
        x = torch.cat([x, cond], dim=1) #add t1 image as conditioning
        #print(f"cat x with cond from c1 = x shape: {x.shape}")
        s1 = torch.cat([s1, f1], dim=1)
        #print(f"cat s1 with f1 from c1 = s1 shape: {s1.shape}")
        s2, x = self.e2(x, t_embed)
        #print(f"self.e2 = s2 shape: {s2.shape}, x shape: {x.shape}")
        f2, cond = self.c2(cond, t_embed)
        #print(f"self.c2 = f2 shape: {f2.shape}, cond shape: {cond.shape}")
        x = torch.cat([x, cond], dim=1)
        #print(f"cat x with cond from c2 = x shape: {x.shape}")
        s2 = torch.cat([s2, f2], dim=1)
        #print(f"cat s2 with f2 from c2 = s2 shape: {s2.shape}")
        # print(x.shape)
        s3, x = self.e3(x, t_embed)
        f3, cond = self.c3(cond, t_embed)
        x = torch.cat([x, cond], dim=1)
        s3 = torch.cat([s3, f3], dim=1)
        s4, x = self.e4(x, t_embed)
        f4, cond = self.c4(cond, t_embed)
        x = torch.cat([x, cond], dim=1)
        s4 = torch.cat([s4, f4], dim=1)

        # bottleneck
        x = self.b1(x, t_embed)
        x = self.b2(x, t_embed)

        # decoder
        x = self.d1(x, s4, t_embed)
        x = self.d2(x, s3, t_embed)
        x = self.d3(x, s2, t_embed)
        #print(x.shape, s1.shape)
        x = self.d4(x, s1, t_embed)

        # output
        output_eps = self.output_eps(x)
        output_eps = output_eps[:, :, :H, :W]
        #output_seg = self.output_seg(x)
        #output_seg = output_seg[:, :, :H, :W]
        return output_eps#, output_seg