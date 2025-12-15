
import torch.nn as nn
from typing import List

class Autoencoder(nn.Module):
    """
    Simple autoencoder for dimensionality reduction with distance preservation.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        assert len(hidden_dims) > 0, "Ensure that there is some hidden layer"

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )

        for i in range(len(hidden_dims)-1):
          self.encoder.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
          self.encoder.append(nn.ReLU())

        self.encoder.append(nn.Linear(hidden_dims[-1], latent_dim))

        # Reverse the encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU()
        )

        for i in range(len(hidden_dims)-1, 0, -1):
          self.decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
          self.decoder.append(nn.ReLU())

        self.decoder.append(nn.Linear(hidden_dims[0], input_dim))

    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space to input space"""
        return self.decoder(z)

    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        return self.decode(z)