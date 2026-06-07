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
            sigma_list=[0.5, 1.0, 2.0],   # was [1.0, 5.0, 10.0]
            # Laminar Poiseuille flow is smooth (polynomial u=2(1-r²/R²)).
            # High σ=10 allowed high-frequency noise that made |dv/dy|≈0.4 and
            # |dw/dz|≈0.4, driving ∇·u=0.32 even when shape was perfect (R²=0.999).
            # σ=[0.5,1,2] limits spatial frequencies to ~4 rad/unit (physical),
            # which is more than enough for the quadratic Poiseuille profile.
            # This forces smooth, low-frequency v and w → small dv/dy, dw/dz → ∇·u≈0.
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
        """
        Koordinatları [-1, 1] aralığına güvenli (out-of-place) bir şekilde normalize eder.
        Bu yöntem PyTorch'un türev zincirini bozmaz.
        """
        # Koordinatları parçalarına ayır (Slicing)
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]
        
        # Normalizasyon işlemlerini yeni değişkenlerde yap (Autograd-safe)
        # x: [0, L] -> [-1, 1]
        x_n = (x / self.length) * 2.0 - 1.0
        # y, z: [-R, R] -> [-1, 1]
        y_n = y / self.radius
        z_n = z / self.radius
        
        # Parçaları yeni bir tensör olarak birleştir (torch.cat)
        coords_norm = torch.cat([x_n, y_n, z_n], dim=1)
        
        # Makalendeki Fourier Embedding katmanına besle ve MLP'den geçir
        return self.net(self.embedding(coords_norm))