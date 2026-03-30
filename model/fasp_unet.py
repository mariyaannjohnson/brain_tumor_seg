"""
fasp_unet.py  —  Architecture only (no training, no loss, no dataset)
Extracted from FASP-UNet v5 training code for inference use.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# GPU-SAFE HAAR WAVELET
# ============================================================================

class HaarWaveletGPU(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5,-0.5], [ 0.5,  0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5,  0.5]], dtype=torch.float32)
        hh = torch.tensor([[ 0.5,-0.5], [-0.5,  0.5]], dtype=torch.float32)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        filters = self.filters.to(x.device)
        outputs = []
        for c in range(C):
            inp    = x[:, c:c+1, :, :].contiguous()
            coeffs = F.conv2d(inp, filters, stride=2, padding=0)
            outputs.append(coeffs)
        return torch.cat(outputs, dim=1)


# ============================================================================
# CORE BUILDING BLOCKS
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn        = nn.BatchNorm2d(out_channels)
        self.relu      = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class EfficientCrossDomainAttention(nn.Module):
    def __init__(self, spatial_channels, frequency_channels, reduction_factor=4):
        super().__init__()
        self.reduction_factor = reduction_factor
        reduced = max(frequency_channels // 2, 4)
        self.query_conv = nn.Conv2d(spatial_channels,   reduced,            1)
        self.key_conv   = nn.Conv2d(frequency_channels, reduced,            1)
        self.value_conv = nn.Conv2d(frequency_channels, frequency_channels, 1)
        self.scale      = reduced ** -0.5
        self.out_conv   = nn.Conv2d(frequency_channels, frequency_channels, 1)

    def forward(self, spatial_feat, freq_feat):
        B, C_f, H_f, W_f = freq_feat.shape
        B, C_s, H_s, W_s = spatial_feat.shape
        th = max(1, H_f // self.reduction_factor)
        tw = max(1, W_f // self.reduction_factor)
        sf = F.interpolate(spatial_feat, (th, tw), mode='bilinear', align_corners=False) \
             if (H_s != th or W_s != tw) else spatial_feat
        ff = F.interpolate(freq_feat, (th, tw), mode='bilinear', align_corners=False) \
             if (H_f != th or W_f != tw) else freq_feat
        q    = self.query_conv(sf).reshape(B, -1, th * tw).permute(0, 2, 1)
        k    = self.key_conv(ff).reshape(B, -1, th * tw)
        v    = self.value_conv(ff).reshape(B, C_f, th * tw).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C_f, th, tw)
        out  = self.out_conv(out)
        if H_s != th or W_s != tw:
            out = F.interpolate(out, (H_s, W_s), mode='bilinear', align_corners=False)
        return out


class SimplifiedDualDomainInput(nn.Module):
    def __init__(self, in_channels=4, spatial_channels=32, frequency_channels=8):
        super().__init__()
        self.spatial_conv = DepthwiseSeparableConv(in_channels, spatial_channels)
        self.dwt          = HaarWaveletGPU()
        self.freq_conv    = nn.Conv2d(in_channels * 4, frequency_channels, 1)
        self.freq_bn      = nn.BatchNorm2d(frequency_channels)
        self.freq_relu    = nn.ReLU(inplace=True)
        self.upsample     = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attention    = EfficientCrossDomainAttention(
            spatial_channels, frequency_channels, reduction_factor=4)

    def forward(self, x):
        spatial_feat  = self.spatial_conv(x)
        freq_coeffs   = self.dwt(x)
        freq_feat     = self.freq_relu(self.freq_bn(self.freq_conv(freq_coeffs)))
        freq_feat     = self.upsample(freq_feat)
        weighted_freq = self.attention(spatial_feat, freq_feat)
        return torch.cat([spatial_feat, weighted_freq], dim=1)   # [B, 40, H, W]


class SimpleEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels,  out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x        = self.conv1(x)
        features = self.conv2(x)
        return features, self.pool(features)


class UncertaintyBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=512, dropout_rate=0.3, threshold=0.5):
        super().__init__()
        self.threshold        = threshold
        self.conv             = DepthwiseSeparableConv(in_channels, out_channels)
        self.dropout          = nn.Dropout2d(p=dropout_rate)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features      = self.conv(x)
        dropped       = self.dropout(features)
        uncertainty   = self.uncertainty_head(dropped)
        sparsity_mask = (uncertainty > self.threshold).float()
        return features, uncertainty, sparsity_mask


class SparseDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv     = DepthwiseSeparableConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, uncertainty_map, sparsity_mask):
        x = self.upsample(x)
        B, C_x, H_x, W_x = x.shape
        if skip.shape[2] != H_x or skip.shape[3] != W_x:
            skip = F.interpolate(skip, (H_x, W_x), mode='bilinear', align_corners=False)
        unc_r  = F.interpolate(uncertainty_map, (H_x, W_x), mode='bilinear', align_corners=False)
        spar_r = F.interpolate(sparsity_mask,   (H_x, W_x), mode='bilinear', align_corners=False)
        weighted_skip = skip * (1 - unc_r)
        x_conv = self.conv(torch.cat([x, weighted_skip], dim=1))
        return x_conv * spar_r + x * (1 - spar_r)


class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(in_channels // 2, 4)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + mid, in_channels, 1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        return self.refine_conv(torch.cat([x, edge], dim=1))


# ============================================================================
# FASP-UNet  (inference mode — deep_supervision always False)
# ============================================================================

class FASPUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, deep_supervision=False):
        super().__init__()
        # deep_supervision is kept as a param for API compatibility,
        # but during inference it is always False (forward never returns ds outputs).
        self.deep_supervision = deep_supervision

        self.input_module   = SimplifiedDualDomainInput(
            in_channels=4, spatial_channels=32, frequency_channels=8)

        self.enc1       = SimpleEncoderBlock(40,  64)
        self.enc2       = SimpleEncoderBlock(64,  128)
        self.enc3       = SimpleEncoderBlock(128, 256)
        self.bottleneck = UncertaintyBottleneck(256, 512, dropout_rate=0.3, threshold=0.5)

        self.dec3 = SparseDecoderBlock(512, 256, 256)
        self.dec2 = SparseDecoderBlock(256, 128, 128)
        self.dec1 = SparseDecoderBlock(128,  64,  64)
        self.dec0 = SparseDecoderBlock( 64,  40,  32)

        self.boundary_refine = BoundaryRefinementModule(32)
        self.output_conv     = nn.Conv2d(32, num_classes, 1)

        # Deep supervision heads — weights are loaded from checkpoint
        # even though they are never used at inference time.
        if self.deep_supervision:
            self.ds_out3 = nn.Conv2d(256, num_classes, 1)
            self.ds_out2 = nn.Conv2d(128, num_classes, 1)
            self.ds_out1 = nn.Conv2d( 64, num_classes, 1)

    def forward(self, x):
        x0 = self.input_module(x)

        skip1, x1 = self.enc1(x0)
        skip2, x2 = self.enc2(x1)
        skip3, x3 = self.enc3(x2)

        features, uncertainty, sparsity_mask = self.bottleneck(x3)

        x_dec3 = self.dec3(features, skip3, uncertainty, sparsity_mask)
        x_dec2 = self.dec2(x_dec3,   skip2, uncertainty, sparsity_mask)
        x_dec1 = self.dec1(x_dec2,   skip1, uncertainty, sparsity_mask)
        x_dec0 = self.dec0(x_dec1,   x0,    uncertainty, sparsity_mask)

        x_dec0    = F.interpolate(x_dec0, size=(240, 240),
                                  mode='bilinear', align_corners=False)
        x_refined = self.boundary_refine(x_dec0)
        logits    = self.output_conv(x_refined)

        # Always return (logits, uncertainty) — same signature app.py expects
        return logits, uncertainty