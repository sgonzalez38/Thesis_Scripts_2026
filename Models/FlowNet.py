import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# CBAM - it is Defined but it is not used in this implementation, you can enable 
# it by setting use_cbam=True when instantiating the PyramidEncoder
# ------------------------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(1, channels // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1)
        )

        pad = (kernel_size - 1) // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=pad)

    def forward(self, x):
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ch

        # Spatial attention
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        sp = torch.cat([avg, mx], dim=1)
        sp = torch.sigmoid(self.spatial(sp))

        return x * sp

# -----------------------
# Convolutional Block
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_gn=True):
        super().__init__()
        padding = (kernel_size - 1) // 2

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]

        if use_gn:
            groups = min(8, max(1, out_channels // 16))
            layers.append(nn.GroupNorm(groups, out_channels))

        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# -----------------------
# Encoder 
# -----------------------
class PyramidEncoder(nn.Module):
    def __init__(self, use_cbam=False):
        super().__init__()
        self.use_cbam = use_cbam
        self.conv1 = ConvBlock(1, 64, 7, 2)
        self.conv2 = ConvBlock(64, 128, 5, 2)
        self.conv3 = ConvBlock(128, 256, 5, 2)
        self.conv3_1 = ConvBlock(256, 256)

        #If use_cbam is True, we add CBAM modules after each convolutional block in the encoder
        if self.use_cbam:
            self.cbam1 = CBAM(64, reduction=8)
            self.cbam2 = CBAM(128, reduction=8)
            self.cbam3 = CBAM(256, reduction=8)

    def forward(self, x):
        f1 = self.conv1(x)
        if self.use_cbam:
            f1 = self.cbam1(f1)
            
        f2 = self.conv2(f1)
        if self.use_cbam:
            f2 = self.cbam2(f2)
            
        f3 = self.conv3_1(self.conv3(f2))
        if self.use_cbam:
            f3 = self.cbam3(f3)
            
        return f1, f2, f3

# -----------------------
# Correlation calculation
# -----------------------
class Correlation(nn.Module):
    def __init__(self, max_disp=4, normalize=False):
        super().__init__()
        self.max_disp = max_disp
        self.normalize = normalize

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        D = self.max_disp
        K = 2 * D + 1 # Kernel size for correlation

        f2_pad = F.pad(f2, (D, D, D, D), mode="replicate") # (B, C, H+2D, W+2D)

        f2_unfold = F.unfold(f2_pad, kernel_size=K, stride=1) # (B, C*K*K, H*W)
        f2_unfold = f2_unfold.view(B, C, K*K, H, W) # (B, C, K*K, H, W)

        f1 = f1.unsqueeze(2)
        corr = (f1 * f2_unfold).sum(1) # (B, K*K, H, W)

        if self.normalize:
            f1_n = torch.norm(f1, dim=1).clamp(min=1e-4)
            f2_n = torch.norm(f2_unfold, dim=1).clamp(min=1e-4)
            corr = corr / (f1_n * f2_n)

        return corr

# ---------------------------------
# Warp function for feature warping
# ---------------------------------
def warp(x, flow):
    B, C, H, W = x.shape
    device = x.device # We need to ensure the grid is on the same device as the input tensor

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    ) #This creates a grid of pixel coordinates (H, W)

    grid = torch.stack((xx, yy), 0).float()
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    vgrid = grid + flow # This adds the flow to the pixel coordinates to get the new sampling locations

    vgrid[:, 0] = 2 * vgrid[:, 0] / (W - 1) - 1
    vgrid[:, 1] = 2 * vgrid[:, 1] / (H - 1) - 1

    vgrid = vgrid.permute(0, 2, 3, 1)

    return F.grid_sample(x, vgrid, align_corners=True, padding_mode='border') 

# ----------------------------------------------------------------------
# Flow Estimator: A simple CNN to estimate flow from correlation volumes
# ----------------------------------------------------------------------
class FlowEstimator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 256),
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, 2, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------
# Refine Head for full resolution refinement
# ------------------------------------------
class RefineHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return self.conv3(torch.cat([out1, out2], dim=1))

# -------------------------------------------------------------
# FlowNet Pyramidal: Main architecture combining all components
# -------------------------------------------------------------
class FlowNetPyramidal(nn.Module):
    def __init__(self, max_disp=4, normalize_corr=True, use_cbam=False):
        super().__init__()
        # The use_cbam flag allows us to enable or disable the CBAM modules in the encoder.
        self.encoder = PyramidEncoder(use_cbam=use_cbam)
        self.corr = Correlation(max_disp, normalize_corr)

        corr_ch = (2 * max_disp + 1) ** 2

        self.est3 = FlowEstimator(corr_ch)
        self.est2 = FlowEstimator(corr_ch + 128 + 2)
        self.est1 = FlowEstimator(corr_ch + 64 + 2)

        self.refine = RefineHead(132)

    def flow_to_heatmap(self, flow):
        # This function converts the flow vectors into a heatmap by calculating the magnitude of the flow at each pixel.
        return torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2)

    def forward(self, ref, samp):
        f_ref1, f_ref2, f_ref3 = self.encoder(ref)
        f_sam1, f_sam2, f_sam3 = self.encoder(samp)

        # Level 3 - Coarsest level
        c3 = self.corr(f_ref3, f_sam3)
        flow3 = self.est3(c3)

        # Level 2 - Mid level
        flow3_up = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        sam2_w = warp(f_sam2, flow3_up)
        c2 = self.corr(f_ref2, sam2_w)

        flow2 = flow3_up + self.est2(torch.cat([c2, f_ref2, flow3_up], dim=1))

        # Level 1 - Full resolution
        flow2_up = F.interpolate(flow2, scale_factor=2, mode='bilinear', align_corners=True) * 2
        sam1_w = warp(f_sam1, flow2_up)
        c1 = self.corr(f_ref1, sam1_w)

        flow1 = flow2_up + self.est1(torch.cat([c1, f_ref1, flow2_up], dim=1))

        # Full resolution refinement
        flow_up = F.interpolate(flow1, scale_factor=2, mode='bilinear', align_corners=True) * 2
        samp_w = warp(samp, flow_up)

        f_sam1_w_up = F.interpolate(sam1_w, scale_factor=2, mode='bilinear', align_corners=True)
        f_ref1_up = F.interpolate(f_ref1, scale_factor=2, mode='bilinear', align_corners=True)

        refine_inp = torch.cat([ref, samp_w, flow_up, f_ref1_up, f_sam1_w_up], dim=1)

        flow_final = flow_up + self.refine(refine_inp)

        # Heatmaps
        heat_l3 = self.flow_to_heatmap(flow2)
        heat_l2 = self.flow_to_heatmap(flow1)
        heat_l1 = self.flow_to_heatmap(flow_final)

        return {
            "flow_l3": flow2,
            "flow_l2": flow1,
            "flow_l1": flow_final,
            "heatmap_l3": heat_l3,
            "heatmap_l2": heat_l2,
            "heatmap_l1": heat_l1
        }