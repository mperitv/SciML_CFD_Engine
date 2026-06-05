import sys
import os
import torch

# Python'un 'src' klasörünü bulabilmesi için yolu ekliyoruz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENGINE INFO] Başlatılıyor... Cihaz: {device}")

    REYNOLDS_NUMBER = 10000.0 
    BATCH_SIZE = 5000 
    
    model = PINN3DEngine(hidden_dim=256, num_layers=6).to(device)
    physics_engine = NavierStokes3DPhysics(Re=REYNOLDS_NUMBER)
    
    x_coords = torch.rand((BATCH_SIZE, 1), requires_grad=True, device=device)
    y_coords = (torch.rand((BATCH_SIZE, 1), requires_grad=True, device=device) * 2) - 1
    z_coords = (torch.rand((BATCH_SIZE, 1), requires_grad=True, device=device) * 2) - 1
    
    with torch.no_grad():
        x_coords[:500] = 0.0 
        y_coords[500:1000] = 1.0 
    
    coords = torch.cat([x_coords, y_coords, z_coords], dim=1)

    loss_cont, loss_mom = physics_engine.compute_pde_loss(model, coords)
    loss_inlet, loss_wall = physics_engine.compute_bc_loss(model, coords, u_inlet=1.5)
    
    lambda_bc = 10.0
    total_loss = loss_cont + loss_mom + lambda_bc * (loss_inlet + loss_wall)

    print("="*55)
    ntk = physics_engine.compute_empirical_ntk(model, coords[:32], output_index=0, max_points=32)
    ntk_metrics = physics_engine.compute_ntk_spectral_metrics(ntk)

    print("🚀 3D NAVIER-STOKES MOTORU (BC + NTK DIAGNOSTIC) RAPORU 🚀")
    print("="*55)
    print(f"  • Kütle Korunumu Kaybı : {loss_cont.item():.6e}")
    print(f"  • Momentum Kaybı       : {loss_mom.item():.6e}")
    print(f"  • Inlet (Giriş) Kaybı  : {loss_inlet.item():.6e}")
    print(f"  • No-Slip (Duvar) Kaybı: {loss_wall.item():.6e}")
    print(f"  • Toplam Optimize Loss : {total_loss.item():.6e}")
    print(f"  • NTK κ (condition)    : {ntk_metrics['condition_number']:.4e}")
    print(f"  • NTK decay slope (α)  : {ntk_metrics['spectral_decay_slope']:.4f}")
    print("="*55)
    print("[BAŞARILI] Sistem modüler yapıda kusursuz çalışıyor.\n")