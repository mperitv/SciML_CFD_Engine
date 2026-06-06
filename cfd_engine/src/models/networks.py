import torch
import torch.nn as nn
from .embeddings import MultiScaleFourierEmbedding

class PINN3DEngine(nn.Module):
    """3D Navier-Stokes Çözücü Derin Sinir Ağı"""
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        
        self.embedding = MultiScaleFourierEmbedding(
            in_features=3,
            sigma_list=[1.0, 5.0, 10.0],
            frequencies_per_scale=64
        )
        
        layers = []
        in_dim = self.embedding.out_features
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
            
        layers.append(nn.Linear(hidden_dim, 4)) # Çıkış: u, v, w, p
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(self.embedding(coords))