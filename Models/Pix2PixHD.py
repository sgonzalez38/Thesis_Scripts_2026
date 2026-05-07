import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

#----------------------------------------------
# Residual Block 
#----------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=lambda c: nn.InstanceNorm2d(c, affine=True)):
        super().__init__()
        self.cbam = CBAM(dim, ratio=8)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            self.cbam
        )

    def forward(self, x):
        return x + self.block(x)


def upsample_conv(in_c, out_c, norm_layer=lambda c: nn.InstanceNorm2d(c, affine=True)):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        norm_layer(out_c),
        nn.ReLU(True)
    )


#-------------------------------
# CBAM implementation 
#-------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        hidden_planes = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7" #This because the padding is hardcoded for these sizes
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, return_attention=False):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()
        self.return_attention = return_attention

    def forward(self, x):
        ca_map = self.ca(x)
        x_ca = x * ca_map
        sa_map = self.sa(x_ca)
        out = x_ca * sa_map
        if self.return_attention:
            return out, ca_map, sa_map
        return out


#------------------------------------------------------------------------------
# Global Generator (G1)
# - forward_features(x) -> feature map (channels = ngf)
# - forward(x)         -> image (1 channel) for standalone use
#------------------------------------------------------------------------------

class GlobalGenerator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        ngf=64, #ngf is the number of filters in the first conv layer; it doubles with each downsampling
        n_downsampling=4,
        n_blocks=9,
        norm_layer=lambda c: nn.InstanceNorm2d(c, affine=True)
    ):
        super().__init__()

        # encoder initial
        self.enc_init = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        # encoder downsampling
        enc_down = []
        for i in range(n_downsampling):
            mult = 2 ** i
            enc_down.extend([
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ])
        self.enc_down = nn.Sequential(*enc_down)

        # resnet blocks
        mult = 2 ** n_downsampling
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResnetBlock(ngf * mult, norm_layer))
        self.res_blocks = nn.Sequential(*res_blocks)

        # decoder upsampling -> returns features with channels = ngf
        dec_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            dec_up.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ngf * mult, ngf * mult // 2, kernel_size=3, padding=1),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True)
            ])
        self.dec_up = nn.Sequential(*dec_up)

        # final conv to map features -> image
        self.final_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh()
        )

        self.ngf = ngf

    def forward_features(self, x):
        """
        Returns internal feature map (channels == ngf).
        Use this when fusing with local enhancers.
        """
        x = self.enc_init(x)
        x = self.enc_down(x)
        x = self.res_blocks(x)
        x = self.dec_up(x)   # result has channels == ngf
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        out = self.final_conv(feat)
        return out


#------------------------------------------------------------------------------
# Local Enhancer (G2)
# - builds pyramid; **coarsest scale** is computed so that G1 receives
#   the same resolution it was trained with (see comment below).
# - fuses global features and local upsampled features by concatenation
#   and projects with 1x1 conv.
#-------------------------------------------------------------------------------

class LocalEnhancer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        ngf=32,
        n_local_enhancers=1, # Adjusted to 1 for 256px input, since G1 is trained on 128px.
        n_blocks=9, #Number of ResNet blocks in the local enhancer; can be tuned for better performance.
        norm_layer=lambda c: nn.InstanceNorm2d(c, affine=True),
        global_generator=None
    ):
        super().__init__()

        # 1. Global Generator (G1)
        # If you have a pretrained G1 that was trained on 128px images, we need to ensure 
        # that the coarsest scale of the local enhancer also processes 128px inputs.
        if global_generator is not None:
            self.g1 = global_generator
            self.g1_ngf = getattr(global_generator, 'ngf', 64) 
        else:
            self.g1_ngf = ngf * (2 ** n_local_enhancers)
            self.g1 = GlobalGenerator(
                in_channels=in_channels,
                out_channels=out_channels,
                ngf=self.g1_ngf,
                n_downsampling=3,
                n_blocks=3,
                norm_layer=norm_layer
            )

        # 2. Module list initialization
        self.local_down = nn.ModuleList()
        self.local_up = nn.ModuleList()
        self.local_cbam = nn.ModuleList()
        self.fuse_convs = nn.ModuleList() 

        local_up_channels = []

        # Construction of local enhancers
        for i in range(n_local_enhancers):
            mult = 2 ** (n_local_enhancers - i)

            #Local Downsampling
            down = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, ngf * mult, kernel_size=7),
                norm_layer(ngf * mult),
                nn.ReLU(True),
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            )
            self.local_down.append(down)

            # Resnet blocks + Upsampling
            blocks = [ResnetBlock(ngf * mult * 2, norm_layer) for _ in range(n_blocks)]
            up = nn.Sequential(
                *blocks,
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, padding=1),
                norm_layer(ngf * mult),
                nn.ReLU(True)
            )
            self.local_up.append(up)
            
            # CBAM module after upsampling
            self.local_cbam.append(CBAM(ngf * mult, ratio=8))
            local_up_channels.append(ngf * mult)

        # 3. Fusion layers 
        seq = list(reversed(local_up_channels))
        for idx, local_ch in enumerate(seq):
            if idx == 0:
                in_ch = self.g1_ngf + local_ch
            else:
                in_ch = ngf + local_ch
            
            self.fuse_convs.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, ngf, kernel_size=3),
                norm_layer(ngf),
                nn.ReLU(True)
            ))

        # Final layer to map fused features to output image (1 channel)
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh()
        )

        self.ngf = ngf
        self.n_local_enhancers = n_local_enhancers

    def forward(self, x):
        inputs = [x]
        # We create a pyramid of images for G1 and local levels
        for _ in range(len(self.local_down) + 1):
            inputs.append(F.avg_pool2d(inputs[-1], kernel_size=2))

        # We take features from G1 at the coarsest scale (128px)
        g_feat = self.g1.forward_features(inputs[-1]) 

        output = g_feat
        
        # We process the fusion hierarchy
        for idx in range(len(self.local_down)):
            level = len(self.local_down) - 1 - idx
            
            # Local branch (Escala 256px)
            local_feat = self.local_down[level](inputs[level])
            local_up = self.local_up[level](local_feat)
            local_up = self.local_cbam[level](local_up)

            # Asert that the spatial dimensions match before concatenation; if not, we resize the output from the 
            # global branch to match the local upsampled features.
            if output.shape[2:] != local_up.shape[2:]:
                output = F.interpolate(output, size=local_up.shape[2:], mode="bilinear", align_corners=False)

            # Feature concatenation and fusion
            fused = torch.cat([output, local_up], dim=1)
            output = self.fuse_convs[idx](fused)

        return self.final(output)


#-----------------------------------------------------------------------------
# Discriminators
# - GroupNorm was used previously in proposals; to remain robust to small
#   spatial sizes we provide a default norm that works with tiny spatial sizes.
# - The discriminator returns a list of intermediate activations (features)
#   for feature matching; MultiscaleDiscriminator skips too-small scales.
#------------------------------------------------------------------------------


def default_disc_norm(c):
    # GroupNorm with 1 group is robust for tiny spatial sizes and small batches
    return nn.GroupNorm(1, c)


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64, n_layers=4):
        super().__init__()
        kw = 4
        padw = 1
        layers = []

        # 1. First layer: No normalization, only spectral normalization (SN) to mantain GAN stability; stride=2 for downsampling
        layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        # 2. Intermediate layers (Downsampling)
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        # 3. This layer Maintains resolution (stride=1) for edge refinement; also applies spectral normalization for stability
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        # 4. Final layer: Mapping to 1-channel prediction (PatchGAN)
        # Also applies spectral_norm to maintain the Lipschitz constraint across the entire D
        layers.append(spectral_norm(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64, n_layers=4, n_discriminators=3):
        super().__init__()
        self.n_discriminators = n_discriminators
        self.discriminators = nn.ModuleList()
        for _ in range(n_discriminators):
            self.discriminators.append(NLayerDiscriminator(in_channels, ndf, n_layers))
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        input_down = x
        for disc in self.discriminators:
            h, w = input_down.shape[2], input_down.shape[3]
            if h < 4 or w < 4:
                break
            results.append(disc(input_down))
            input_down = self.downsample(input_down)
        return results
