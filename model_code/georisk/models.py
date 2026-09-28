import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    return 8 if channels % 8 == 0 else 4


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, initial_channels=64):
        super().__init__()
        c = initial_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.classifier(d1)
        embedding = F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)
        return logits, embedding


def _interpolated_position_embedding(position_embedding, height, width):
    patch_tokens = position_embedding[:, 1:]
    source_size = int(math.sqrt(patch_tokens.shape[1]))
    patch_tokens = patch_tokens.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens, size=(height, width), mode="bicubic", align_corners=False
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return position_embedding[:, :1], patch_tokens


class DOFADynamicEncoder(nn.Module):
    def __init__(self, source_dir, weights_path, band_descriptors):
        super().__init__()
        source_dir = Path(source_dir)
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        from downstream_tasks.geobench_segmentation.model import vit_base_patch16

        self.backbone = vit_base_patch16(img_size=224, drop_path_rate=0.0)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        checkpoint = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.backbone.load_state_dict(checkpoint, strict=False)
        self.band_descriptors = list(band_descriptors)
        self.out_indices = set(self.backbone.out_indices)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        wavelengths = torch.tensor(self.band_descriptors, device=x.device, dtype=x.dtype)
        tokens, _ = self.backbone.patch_embed(x, wavelengths)
        patch_size = int(self.backbone.patch_embed.kernel_size)
        grid_height = x.shape[-2] // patch_size
        grid_width = x.shape[-1] // patch_size
        cls_position, patch_position = _interpolated_position_embedding(
            self.backbone.pos_embed, grid_height, grid_width
        )
        tokens = tokens + patch_position
        cls_tokens = (self.backbone.cls_token + cls_position).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        outputs = []
        for index, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            if index in self.out_indices:
                feature = tokens[:, 1:].reshape(
                    tokens.shape[0], grid_height, grid_width, tokens.shape[-1]
                )
                outputs.append(feature.permute(0, 3, 1, 2).contiguous())
        embedding = F.adaptive_avg_pool2d(outputs[-1], 1).flatten(1)
        return outputs, embedding


class FeaturePyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rescale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(channels, channels, 2, stride=2),
                ),
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.Identity(),
                nn.MaxPool2d(2, stride=2),
            ]
        )

    def forward(self, features):
        return [operation(feature) for operation, feature in zip(self.rescale, features)]


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(1, 2, 3, 6)):
        super().__init__()
        self.scales = scales
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.GroupNorm(_groups(out_channels), out_channels),
                    nn.ReLU(inplace=True),
                )
                for scale in scales
            ]
        )
        self.bottleneck = ConvBlock(in_channels + len(scales) * out_channels, out_channels)

    def forward(self, x):
        pooled = [x]
        for projection in self.projections:
            value = projection(x)
            pooled.append(F.interpolate(value, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(pooled, dim=1))


class UPerNetHead(nn.Module):
    def __init__(self, in_channels=768, channels=128, num_classes=2):
        super().__init__()
        self.ppm = PyramidPooling(in_channels, channels)
        self.lateral = nn.ModuleList(
            [nn.Conv2d(in_channels, channels, 1) for _ in range(3)]
        )
        self.fpn = nn.ModuleList([ConvBlock(channels, channels) for _ in range(3)])
        self.fuse = ConvBlock(channels * 4, channels)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features, output_size):
        laterals = [layer(feature) for layer, feature in zip(self.lateral, features[:3])]
        laterals.append(self.ppm(features[3]))
        for index in range(3, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        outputs = [layer(value) for layer, value in zip(self.fpn, laterals[:3])]
        outputs.append(laterals[3])
        target_size = outputs[0].shape[-2:]
        outputs = [
            value if value.shape[-2:] == target_size else F.interpolate(
                value, size=target_size, mode="bilinear", align_corners=False
            )
            for value in outputs
        ]
        logits = self.classifier(self.dropout(self.fuse(torch.cat(outputs, dim=1))))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DOFAUPerNet(nn.Module):
    def __init__(self, source_dir, weights_path, band_descriptors, num_classes):
        super().__init__()
        self.encoder = DOFADynamicEncoder(source_dir, weights_path, band_descriptors)
        self.neck = FeaturePyramid(768)
        self.decoder = UPerNetHead(768, 128, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features, embedding = self.encoder(x)
        logits = self.decoder(self.neck(features), x.shape[-2:])
        return logits, embedding


def build_model(model_name, dataset_info, root, model_config):
    if model_name == "unet":
        return UNet(
            dataset_info["channels"],
            dataset_info["classes"],
            model_config.get("initial_channels", 64),
        )
    return DOFAUPerNet(
        Path(root) / "third_party" / "DOFA",
        Path(root) / "models" / model_config["weights"],
        dataset_info["dofa_band_descriptors"],
        dataset_info["classes"],
    )

