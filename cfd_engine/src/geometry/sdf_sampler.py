import torch
import math

class PipeGeometrySampler:
    """
    Mesh-free point sampler for 3D cylindrical pipe geometry.
    Flow direction is aligned with the X-axis. Y and Z axes define the cross-section.
    """
    def __init__(self, radius: float = 1.0, length: float = 5.0):
        self.radius = radius
        self.length = length

    def sample_interior(self, num_points: int, device: torch.device) -> torch.Tensor:
        """Generates random collocation points inside the pipe domain for PDE evaluation."""
        x = torch.rand((num_points, 1), device=device) * self.length
        
        # FIXED: Using torch.sqrt for tensor operations instead of math.sqrt
        r = torch.sqrt(torch.rand((num_points, 1), device=device)) * self.radius
        theta = torch.rand((num_points, 1), device=device) * 2.0 * math.pi
        
        y = r * torch.cos(theta)
        z = r * torch.sin(theta)
        
        coords = torch.cat([x, y, z], dim=1)
        coords.requires_grad_(True)
        return coords

    def sample_walls(self, num_points: int, device: torch.device) -> torch.Tensor:
        """Generates points on the outer wall surface for No-Slip boundary conditions."""
        x = torch.rand((num_points, 1), device=device) * self.length
        theta = torch.rand((num_points, 1), device=device) * 2.0 * math.pi
        
        y = self.radius * torch.cos(theta)
        z = self.radius * torch.sin(theta)
        
        coords = torch.cat([x, y, z], dim=1)
        coords.requires_grad_(True)
        return coords

    def sample_inlet(self, num_points: int, device: torch.device) -> torch.Tensor:
        """Generates points at the inlet boundary (x = 0)."""
        x = torch.zeros((num_points, 1), device=device)
        
        # FIXED: Using torch.sqrt
        r = torch.sqrt(torch.rand((num_points, 1), device=device)) * self.radius
        theta = torch.rand((num_points, 1), device=device) * 2.0 * math.pi
        
        y = r * torch.cos(theta)
        z = r * torch.sin(theta)
        
        coords = torch.cat([x, y, z], dim=1)
        coords.requires_grad_(True)
        return coords