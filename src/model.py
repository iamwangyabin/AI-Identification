from __future__ import annotations

from pathlib import Path
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models._registry import is_model


def _unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        tensor_values = all(isinstance(value, torch.Tensor) for value in checkpoint.values())
        if tensor_values:
            return checkpoint
        for key in ("state_dict", "model", "model_state_dict", "teacher", "student"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _unwrap_state_dict(value)
    raise ValueError("Unable to locate a tensor state_dict in the checkpoint.")


def _candidate_state_dicts(
    state_dict: dict[str, torch.Tensor],
) -> Iterable[dict[str, torch.Tensor]]:
    prefixes = ("", "module.", "backbone.", "encoder.", "model.", "student.", "teacher.")
    for prefix in prefixes:
        if not prefix:
            yield state_dict
            continue
        trimmed = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if trimmed:
            yield trimmed


class LocalArtifactBranch(nn.Module):
    def __init__(self, channels: int, out_dim: int) -> None:
        super().__init__()
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, out_dim),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        smoothed = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        residual = feature - smoothed
        enhanced = self.enhance(residual)
        avg_descriptor = F.adaptive_avg_pool2d(enhanced, output_size=1).flatten(1)
        max_descriptor = F.adaptive_max_pool2d(enhanced, output_size=1).flatten(1)
        return self.proj(torch.cat([avg_descriptor, max_descriptor], dim=1))


class GlobalStyleBranch(nn.Module):
    def __init__(self, channels: int, out_dim: int) -> None:
        super().__init__()
        hidden_dim = max(channels // 2, out_dim)
        self.style_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, out_dim),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        gate = self.style_gate(feature).unsqueeze(-1).unsqueeze(-1)
        gated = feature * gate
        pooled = F.adaptive_avg_pool2d(gated, output_size=1).flatten(1)
        return self.proj(pooled)


class ConvNeXtForgeryClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "convnext_tiny",
        out_indices: tuple[int, ...] = (1, 2, 3),
        align_dim: int = 192,
        local_dim: int = 128,
        global_dim: int = 128,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        backbone_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.requested_backbone_name = backbone_name
        self.backbone_name = self._resolve_backbone_name(backbone_name)
        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=pretrained and backbone_checkpoint is None,
                features_only=True,
                out_indices=out_indices,
            )
        except Exception as error:
            if pretrained and backbone_checkpoint is None:
                raise RuntimeError(
                    f"Failed to initialize the pretrained backbone '{self.requested_backbone_name}'. "
                    "If your installed timm build does not include this model yet, upgrade timm; "
                    "otherwise provide --backbone-checkpoint /path/to/weights.pt or disable "
                    "pretrained initialization with --no-pretrained."
                ) from error
            raise
        self.out_indices = out_indices

        if backbone_checkpoint:
            self._load_backbone_checkpoint(backbone_checkpoint)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        channels = self.backbone.feature_info.channels()
        self.align_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, align_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(align_dim),
                    nn.GELU(),
                )
                for channel in channels
            ]
        )
        self.local_branches = nn.ModuleList(
            [LocalArtifactBranch(align_dim, local_dim) for _ in channels]
        )
        self.global_branches = nn.ModuleList(
            [GlobalStyleBranch(align_dim, global_dim) for _ in channels]
        )

        fused_dim = len(channels) * (align_dim + local_dim + global_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    @staticmethod
    def _resolve_backbone_name(backbone_name: str) -> str:
        if is_model(backbone_name):
            return backbone_name
        if "." in backbone_name:
            fallback_name = backbone_name.split(".", 1)[0]
            if is_model(fallback_name):
                return fallback_name
        return backbone_name

    def _load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = _unwrap_state_dict(checkpoint)
        last_error: RuntimeError | None = None
        for candidate in _candidate_state_dicts(state_dict):
            try:
                self.backbone.load_state_dict(candidate, strict=False)
                return
            except RuntimeError as error:
                last_error = error
        raise RuntimeError(
            f"Failed to load backbone checkpoint '{checkpoint_path}' into '{self.backbone_name}'."
        ) from last_error

    def train(self, mode: bool = True) -> "ConvNeXtForgeryClassifier":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pyramid = self.backbone(images)

        descriptors: list[torch.Tensor] = []
        for feature, align, local_branch, global_branch in zip(
            pyramid, self.align_layers, self.local_branches, self.global_branches
        ):
            aligned = align(feature)
            pooled = F.adaptive_avg_pool2d(aligned, output_size=1).flatten(1)
            local_descriptor = local_branch(aligned)
            global_descriptor = global_branch(aligned)
            descriptors.append(torch.cat([pooled, local_descriptor, global_descriptor], dim=1))

        fused = torch.cat(descriptors, dim=1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "features": fused,
        }
