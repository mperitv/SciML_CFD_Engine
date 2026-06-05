import sys
import os
import torch
import logging

# Configure standard Python logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics
from src.geometry.sdf_sampler import PipeGeometrySampler

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing CFD Pipeline. Target device: {device}")

    # System Parameters
    REYNOLDS_NUMBER = 5000.0
    PIPE_RADIUS = 0.5
    PIPE_LENGTH = 3.0
    INLET_VELOCITY = 2.0 

    # Initialize Modules
    model = PINN3DEngine(hidden_dim=256, num_layers=6).to(device)
    physics = NavierStokes3DPhysics(Re=REYNOLDS_NUMBER)
    geometry = PipeGeometrySampler(radius=PIPE_RADIUS, length=PIPE_LENGTH)

    # 1. Mesh-Free Sampling
    interior_pts = geometry.sample_interior(num_points=5000, device=device)
    wall_pts = geometry.sample_walls(num_points=1000, device=device)
    inlet_pts = geometry.sample_inlet(num_points=1000, device=device)

    # 2. Compute Physical Losses (PDE)
    loss_cont, loss_mom = physics.compute_pde_loss(model, interior_pts)

    # 3. Compute Boundary Condition Losses
    preds_wall = model(wall_pts)
    u_w, v_w, w_w = preds_wall[:, 0:1], preds_wall[:, 1:2], preds_wall[:, 2:3]
    loss_wall = torch.mean(u_w**2 + v_w**2 + w_w**2)

    preds_inlet = model(inlet_pts)
    u_in, v_in, w_in = preds_inlet[:, 0:1], preds_inlet[:, 1:2], preds_inlet[:, 2:3]
    loss_inlet = torch.mean((u_in - INLET_VELOCITY)**2 + v_in**2 + w_in**2)

    # 4. Total Loss
    lambda_bc = 20.0 
    total_loss = loss_cont + loss_mom + lambda_bc * (loss_wall + loss_inlet)

    # Professional CLI Output
    print("\n" + "="*60)
    print(" 3D PIPE FLOW GEOMETRY TEST RESULTS")
    print("="*60)
    print(f" Interior Points (PDE) : {interior_pts.shape[0]}")
    print(f" Wall Points (No-Slip) : {wall_pts.shape[0]}")
    print(f" Inlet Points          : {inlet_pts.shape[0]}")
    print("-" * 60)
    print(f" Momentum Loss         : {loss_mom.item():.6e}")
    print(f" Wall Loss             : {loss_wall.item():.6e}")
    print(f" Inlet Loss            : {loss_inlet.item():.6e}")
    print(f" Total System Loss     : {total_loss.item():.6e}")
    print("="*60)
    logger.info("Geometry sampling and forward pass executed successfully.\n")