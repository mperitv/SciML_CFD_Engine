import torch
import torch.nn as nn
from .embeddings import MultiScaleFourierEmbedding

class PINN3DEngine(nn.Module):
    """3D Navier-Stokes Çözücü Derin Sinir Ağı"""
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6, length: float = 3.0, radius: float = 0.5):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.length = float(length)
        self.radius = float(radius)
        
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
        coords_norm = coords.clone().detach().requires_grad_(coords.requires_grad)
        coords_norm[:, 0:1] = coords[:, 0:1] / self.length * 2.0 - 1.0
        coords_norm[:, 1:2] = coords[:, 1:2] / self.radius
        coords_norm[:, 2:3] = coords[:, 2:3] / self.radius
        coords_norm = torch.clamp(coords_norm, -1.0, 1.0)
        return self.net(self.embedding(coords_norm))