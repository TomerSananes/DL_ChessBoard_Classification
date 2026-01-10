import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()

        # Image to Patches: Divide 256x256 image into 64 patches of 32x32.
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=32, stride=32)

        # Allows the model to learn the spatial location
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Transformer Layers: Learns relationships between squares.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP Head: Classification head for each of the 64 squares.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 13)  # 13 classes: 6 white pieces, 6 black pieces, 1 empty
        )

    def forward(self, x):
        # Input x shape: [Batch, 3, 256, 256]
        # Patching: [Batch, embed_dim, 8, 8]
        x = self.patch_embed(x)

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