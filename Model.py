import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, num_layers=3):
        super().__init__()

        """
        Hybrid Architecture: Extracts visual features with a CNN, maps them to an 8x8 grid, 
        and uses a Transformer to model global spatial relationships between board squares 
        before final 13-class classification per square.
        """
        self.cnn_backbone = nn.Sequential(
            # Basic features (edges/textures)
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            # Complex patterns (piece contours)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            # Latent space projection & Grid mapping
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # Allows the model to learn the spatial location
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Transformer Layers: Learns relationships between squares.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head: Classification head for each of the 64 squares.
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 13))


    def forward(self, x):
        # Input x shape: [Batch, 3, 256, 256]
        # Patching: [Batch, embed_dim, 8, 8]
        x = self.cnn_backbone(x)

        # Flattening to sequence [Batch, 64, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add spatial knowledge
        x = x + self.pos_embed

        # Process through Transformer
        x = self.transformer(x)  # [Batch, 64, embed_dim]

        # Classify each square
        logits = self.classifier(x)  # [Batch, 64, 13]

        # Reshape to board format: [Batch, 8, 8, 13]
        return logits.view(-1, 8, 8, 13)