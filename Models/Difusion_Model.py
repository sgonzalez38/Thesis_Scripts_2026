import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------
# Sinusoidal Time Embedding 
# -----------------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) o (B,1)
        return: (B, dim)
        """
        t = t.float()

        if t.dim() == 2:
            t = t[:, 0]

        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb

# -----------------------------------------------------------
# CBAM: Convolutional Block Attention Module
# -----------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # We reduce the number of channels in the MLP to save parameters and computation, but it can be adjusted as needed.
        reduced_planes = max(1, in_planes // reduction)
        
        # We use a shared MLP meaning a convolution with kernel size 1
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # 1. Channel attention
        x = x * self.ca(x)
        # 2. Spatial attention
        x = x * self.sa(x)
        return x

# -----------------------------------------------------------
# Residual Block with FiLM conditioning and CBAM
# -----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

        self.norm1 = nn.GroupNorm(groups, out_c)
        self.norm2 = nn.GroupNorm(groups, out_c)

        self.act = nn.SiLU()

        # Scale and shift for FiLM
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_c * 2),
            nn.SiLU()
        )
        
        # CBAM attention module 
        self.cbam = CBAM(out_c, reduction=16)

        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Time conditioning with FiLM
        film = self.time_mlp(t_emb)
        scale, shift = film.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = h * (1 + scale) + shift

        h = self.conv2(h)
        h = self.norm2(h)
        
        # CBAM attention before the final activation
        h = self.cbam(h)
        
        h = self.act(h)

        return h + self.skip(x)

# -----------------------------------------------------------
# Diffusion UNet Conditional Architecture
# -----------------------------------------------------------
class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4

        # Time Embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input: x_t + condition
        self.init_conv = nn.Conv2d(in_channels + 1, base_dim, 3, padding=1)

        # Encoder
        self.down1 = ResidualBlock(base_dim, base_dim * 2, time_dim)
        self.down2 = ResidualBlock(base_dim * 2, base_dim * 4, time_dim)
        self.down3 = ResidualBlock(base_dim * 4, base_dim * 8, time_dim)

        # Bottleneck
        self.mid = ResidualBlock(base_dim * 8, base_dim * 8, time_dim)

        # Decoder
        self.up3 = ResidualBlock(base_dim * 16, base_dim * 4, time_dim)
        self.up2 = ResidualBlock(base_dim * 8, base_dim * 2, time_dim)
        self.up1 = ResidualBlock(base_dim * 4, base_dim, time_dim)

        self.final_conv = nn.Conv2d(base_dim, out_channels, 1)

        self.apply(self._init_weights)

    # -------------------------------------------------------
    # Weight Initialization
    # -------------------------------------------------------
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------
    # Forward
    # -------------------------------------------------------
    def forward(self, x_t, t, condition):
        """
        x_t: (B,1,H,W) Noise at time t
        condition: (B,1,H,W) Speckle image
        t: (B,)
        """
        # This condition ensures that the condition image is resized to match the spatial dimensions of x_t if they are different. 
        # This is important because we will concatenate them along the channel dimension, and they need to have the same height and width.
        if condition.shape[-2:] != x_t.shape[-2:]:
            condition = F.interpolate(condition, size=x_t.shape[-2:], mode="bilinear", align_corners=False)

        t_emb = self.time_mlp(t)

        x = torch.cat([x_t, condition], dim=1)

        x0 = self.init_conv(x)

        # Encoder
        x1 = self.down1(x0, t_emb)
        x1_p = F.avg_pool2d(x1, 2)

        x2 = self.down2(x1_p, t_emb)
        x2_p = F.avg_pool2d(x2, 2)

        x3 = self.down3(x2_p, t_emb)
        x3_p = F.avg_pool2d(x3, 2)

        # Bottleneck
        mid = self.mid(x3_p, t_emb)

        # Decoder
        u3 = F.interpolate(mid, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.up3(u3, t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2, t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.up1(u1, t_emb)

        out = self.final_conv(u1)

        return out