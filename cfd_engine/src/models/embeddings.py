import torch
import torch.nn as nn
import math
from typing import List

class MultiScaleFourierEmbedding(nn.Module):
    """
    Makale Ref: Bölüm 3.1 (Multi-Scale Fourier Feature Embeddings) ve Denklem 1.
    NTK spektrumunu modüle ederek yüksek frekanslı türbülans modlarına erişimi sağlar.
    """
    def __init__(self, in_features: int = 3, sigma_list: List[float] = [1.0, 10.0, 50.0], frequencies_per_scale: int = 42):
        super().__init__()
        self.in_features = in_features
        self.sigma_list = sigma_list
        
        B_components = []
        for sigma in sigma_list:
            b_scale = torch.randn(in_features, frequencies_per_scale) * sigma
            B_components.append(b_scale)
            
        B = torch.cat(B_components, dim=1)
        self.register_buffer('B', B) 
        
        self.out_features = B.shape[1] * 2
        self.sqrt_D = math.sqrt(self.out_features)
        
        self.layer_norm = nn.LayerNorm(self.out_features)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * torch.matmul(v, self.B)
        embedding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) / self.sqrt_D
        return self.layer_norm(embedding)