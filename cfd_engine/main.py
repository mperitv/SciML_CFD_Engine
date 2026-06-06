import torch
import logging
import os
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CFD_Engine_Main")

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics
from src.geometry.sdf_sampler import PipeGeometrySampler
from src.training.trainer import PINNTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing 3D SciML Engine on {device.type.upper()}")

    # Deterministic seed for reproducibility (can be changed per-run)
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Configuration (can be overridden with environment variables)
    EPOCHS = int(os.environ.get("EPOCHS", "1000"))
    REYNOLDS_NUMBER = float(os.environ.get("RE", "50.0"))
    ADAM_EPOCHS = int(os.environ.get("ADAM_EPOCHS", "1600"))
    LBFGS_EPOCHS = int(os.environ.get("LBFGS_EPOCHS", "100"))
    BATCH_INTERIOR = int(os.environ.get("BATCH_INTERIOR", "1000"))
    BATCH_BOUNDARY = int(os.environ.get("BATCH_BOUNDARY", "200"))
    NTK_REG_WEIGHT = float(os.environ.get("NTK_REG_WEIGHT", "1e-4"))
    LAMBDA_POS = float(os.environ.get("LAMBDA_POS", "10.0"))
    LAMBDA_PIN = float(os.environ.get("LAMBDA_PIN", "1.0"))
    LAMBDA_BC = float(os.environ.get("LAMBDA_BC", "10000.0"))
    LAMBDA_TARGET_VEL = float(os.environ.get("LAMBDA_TARGET_VEL", "1000.0"))
    LAMBDA_INLET = float(os.environ.get("LAMBDA_INLET", "100.0"))
    INLET_VELOCITY = float(os.environ.get("INLET_VELOCITY", "1.0"))
    
    # 2. Initialize Core Modules
    radius = float(os.environ.get('RADIUS', '0.5'))
    length = float(os.environ.get('LENGTH', '3.0'))
    model = PINN3DEngine(hidden_dim=256, num_layers=6, length=length, radius=radius).to(device)
    spatial_weight_start = float(os.environ.get('SPATIAL_WEIGHT_START', '1.0'))
    spatial_weight_slope = float(os.environ.get('SPATIAL_WEIGHT_SLOPE', '1.0'))
    physics = NavierStokes3DPhysics(Re=REYNOLDS_NUMBER, spatial_weight_start=spatial_weight_start, spatial_weight_slope=spatial_weight_slope)
    geometry = PipeGeometrySampler(radius=radius, length=length)

    # 3. Setup Trainer
    log_dir = os.environ.get("LOGDIR", "logs")
    trainer = PINNTrainer(
        model=model,
        physics_engine=physics,
        geometry_sampler=geometry,
        device=device,
        lr=float(os.environ.get('LR', '1e-3')),
        lambda_bc=LAMBDA_BC,
        inlet_velocity=INLET_VELOCITY,
        lambda_target_vel=LAMBDA_TARGET_VEL,
        lambda_inlet=LAMBDA_INLET,
        log_dir=log_dir,
        ntk_check_interval=int(os.environ.get('NTK_CHECK_INTERVAL', '1000')),
        ntk_reg_weight=NTK_REG_WEIGHT,
        lambda_pin=LAMBDA_PIN,
        lambda_pos=LAMBDA_POS,
        pump_force_max=float(os.environ.get('PUMP_FORCE_MAX', '1.0')),
        pump_ramp_epochs=int(os.environ.get('PUMP_RAMP_EPOCHS', '200')),
        run_id=os.environ.get('RUN_ID', None),
    )

    # 4. Execute Training (Adam ile kaba taslak, L-BFGS ile pürüzsüzleştirme)
    logger.info(f"Starting simulation. Reynolds Number: {REYNOLDS_NUMBER}")
    history = trainer.train(
        adam_epochs=ADAM_EPOCHS,     # Adam epochs
        lbfgs_epochs=LBFGS_EPOCHS,     # L-BFGS iterations
        batch_size_interior=BATCH_INTERIOR,
        batch_size_boundary=BATCH_BOUNDARY,
    )

    # 5. Save Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    run_id = os.environ.get('RUN_ID', '')
    suffix = f"_{run_id}" if run_id else ""
    save_path = f"checkpoints/pipe_flow_model{suffix}.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model weights saved to {save_path}")

if __name__ == "__main__":
    main()