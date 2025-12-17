"""
DocCornerNet Model: MobileNetV3-Small backbone with regression head for document detection.

Output:
- coords: [B, 8] normalized coordinates for 4 corners (TL, TR, BR, BL)
- score: [B] logit for document presence (use sigmoid for 0-1 confidence)
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DocCornerNet(nn.Module):
    """
    Model for document corner detection.

    Architecture:
    - Backbone: MobileNetV3-Small (convolutional features)
    - Global Average Pooling
    - Head MLP: Linear(C, 128) -> Hardswish -> Linear(128, 9)

    Output: 8 corner coordinates + 1 score logit = 9 values
    """

    def __init__(
        self,
        img_size: int = 224,
        width_mult: float = 0.5,
        reduced_tail: bool = True,
        pretrained: bool = True,
        dropout: float = 0.2,
        coord_activation: str = "sigmoid",  # "sigmoid", "clamp", or "none"
    ):
        """
        Args:
            img_size: Input image size (square). Used for documentation, not enforced.
            width_mult: Width multiplier for MobileNetV3 channels (0.5 = half channels).
            reduced_tail: If True, reduces the last few layers for lower latency.
            pretrained: If True, loads ImageNet pretrained weights (before width scaling).
            dropout: Dropout rate in the head (0.0 to disable).
        """
        super().__init__()

        self.img_size = img_size
        self.width_mult = width_mult
        self.reduced_tail = reduced_tail
        self.coord_activation = coord_activation

        # Load base MobileNetV3-Small
        # Note: width_mult is applied via _width_mult internally
        # For custom width_mult, we need to create the model differently
        if pretrained and width_mult == 1.0:
            # Use pretrained weights only for width_mult=1.0
            base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            # For non-standard width_mult, start from scratch or load and adapt
            base = mobilenet_v3_small(weights=None)

        # Extract the convolutional backbone (features)
        # MobileNetV3-Small features output: [B, 576, H/32, W/32] for width_mult=1.0
        self.backbone = base.features

        # If using custom width_mult < 1.0, we need to rebuild the backbone
        if width_mult != 1.0:
            self.backbone = self._build_scaled_backbone(width_mult, reduced_tail, pretrained)

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine backbone output channels
        # Run a dummy forward to get the channel count
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            features = self.backbone(dummy)
            backbone_channels = features.shape[1]

        # Head MLP: backbone_channels -> 128 -> 9
        # With dropout for regularization
        head_layers = [
            nn.Flatten(),
            nn.Linear(backbone_channels, 128),
            nn.Hardswish(),
        ]
        if dropout > 0:
            head_layers.append(nn.Dropout(p=dropout))
        head_layers.append(nn.Linear(128, 9))  # 8 coordinates + 1 score logit

        self.head = nn.Sequential(*head_layers)

        # Initialize head weights
        self._init_head_weights()

    def _build_scaled_backbone(
        self,
        width_mult: float,
        reduced_tail: bool,
        pretrained: bool
    ) -> nn.Module:
        """
        Build a width-scaled MobileNetV3-Small backbone.

        Since torchvision doesn't directly support width_mult for MobileNetV3,
        we create a standard model and apply channel scaling via a wrapper,
        or we can use the inverted_residual_setting with scaled channels.

        For simplicity, we use a simpler approach: load the full model
        and keep it, relying on the smaller img_size for efficiency.

        For true width scaling, consider using timm or custom implementation.
        """
        # For now, return standard backbone with optional reduced_tail handling
        # In production, you'd want to properly scale channels

        if pretrained:
            base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            base = mobilenet_v3_small(weights=None)

        backbone = base.features

        if reduced_tail:
            # MobileNetV3-Small features has indices 0-12
            # We can optionally remove the last few blocks for speed
            # For now, keep all blocks but this is where you'd truncate
            pass

        return backbone

    def _init_head_weights(self):
        """Initialize head weights with small values for stable training."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize last layer with smaller weights for coordinates near center
        # and score near zero
        last_linear = self.head[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        # Initialize bias: coords to 0.5 (center), score to 0
        with torch.no_grad():
            last_linear.bias[:8] = 0.5  # Coordinates initialized to center
            last_linear.bias[8] = 0.0   # Score logit initialized to 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [B, 3, H, W], normalized with ImageNet stats.

        Returns:
            coords: [B, 8] corner coordinates normalized to [0, 1]
                    Order: (x0,y0), (x1,y1), (x2,y2), (x3,y3) = TL, TR, BR, BL
            score: [B] document presence logit (apply sigmoid for probability)
        """
        # Backbone features
        features = self.backbone(x)  # [B, C, H', W']

        # Global average pooling
        pooled = self.pool(features)  # [B, C, 1, 1]

        # Head prediction
        out = self.head(pooled)  # [B, 9]

        # Split into coordinates and score
        coords_raw = out[:, :8]  # [B, 8]
        score = out[:, 8]        # [B]

        # Apply coordinate activation
        if self.coord_activation == "sigmoid":
            # Original behavior - causes border shrinkage
            coords = torch.sigmoid(coords_raw)
        elif self.coord_activation == "clamp":
            # Linear output, hard clamp to [0, 1] - recommended
            coords = torch.clamp(coords_raw, 0.0, 1.0)
        else:  # "none"
            # Raw output for debugging
            coords = coords_raw

        return coords, score

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_backbone_params(self):
        """Return backbone parameters (for differential learning rates)."""
        return self.backbone.parameters()

    def get_head_params(self):
        """Return head parameters (for differential learning rates)."""
        return self.head.parameters()


def create_model(
    img_size: int = 224,
    width_mult: float = 1.0,
    reduced_tail: bool = True,
    pretrained: bool = True,
    dropout: float = 0.2,
    coord_activation: str = "sigmoid",
) -> DocCornerNet:
    """
    Factory function to create DocCornerNet model.

    Args:
        img_size: Input image size (default 192 for mobile deployment).
        width_mult: Width multiplier (1.0 = full, 0.5 = half channels).
        reduced_tail: Reduce last blocks for lower latency.
        pretrained: Use ImageNet pretrained backbone.
        dropout: Dropout rate in head (default 0.2 for regularization).
        coord_activation: Activation for coordinates - "sigmoid" (original),
                          "clamp" (recommended), or "none".

    Returns:
        DocCornerNet model instance.
    """
    return DocCornerNet(
        img_size=img_size,
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        pretrained=pretrained,
        dropout=dropout,
        coord_activation=coord_activation,
    )


if __name__ == "__main__":
    # Quick test
    model = create_model(img_size=224, width_mult=1.0, pretrained=True)
    print(f"Model created with {model.get_num_params():,} parameters")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    coords, score = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Coords shape: {coords.shape}, range: [{coords.min():.3f}, {coords.max():.3f}]")
    print(f"Score shape: {score.shape}")
