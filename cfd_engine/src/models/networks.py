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
        # HATA DÜZELTME: In-place operation yerine 'torch.cat' kullanarak türev zincirini koruyoruz.
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]
        
        # Koordinatları [-1, 1] aralığına güvenli bir şekilde çekiyoruz
        x_n = (x / self.length) * 2.0 - 1.0
        y_n = y / self.radius
        z_n = z / self.radius
        
        # Yeni bir tensör oluşturarak birleştiriyoruz (Autograd dostu)
        coords_norm = torch.cat([x_n, y_n, z_n], dim=1)
        
        # Makalendeki Fourier Embedding katmanına besliyoruz
        return self.net(self.embedding(coords_norm))