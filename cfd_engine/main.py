import torch
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CFD_Engine_Main")

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics
from src.geometry.sdf_sampler import PipeGeometrySampler
from src.post_processing.visualizer import CFDVisualizer
from src.training.trainer import PINNTrainer

def build_simulation_components(
    reynolds_number: float,
    inlet_velocity: float,
    radius: float,
    length: float,
    run_id: Optional[str] = None,
    log_dir: str = "logs",
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing 3D SciML Engine on {device.type.upper()}")

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    spatial_weight_start = float(os.environ.get('SPATIAL_WEIGHT_START', '1.0'))
    spatial_weight_slope = float(os.environ.get('SPATIAL_WEIGHT_SLOPE', '1.0'))
    outlet_suction_strength = float(os.environ.get('OUTLET_SUCTION_STRENGTH', '5.0'))
    outlet_suction_width = float(os.environ.get('OUTLET_SUCTION_WIDTH', '0.05'))

    geometry = PipeGeometrySampler(radius=float(radius), length=float(length))
    physics = NavierStokes3DPhysics(
        Re=float(reynolds_number),
        spatial_weight_start=spatial_weight_start,
        spatial_weight_slope=spatial_weight_slope,
        pipe_length=float(length),
        outlet_suction_strength=outlet_suction_strength,
        outlet_suction_width=outlet_suction_width,
    )
    model = PINN3DEngine(hidden_dim=256, num_layers=6, length=float(length), radius=float(radius)).to(device)
    trainer = PINNTrainer(
        model=model,
        physics_engine=physics,
        geometry_sampler=geometry,
        device=device,
        lr=float(os.environ.get('LR', '1e-3')),
        lambda_bc=float(os.environ.get('LAMBDA_BC', '10000.0')),
    )
    visualizer = CFDVisualizer(model, device)

    return {
        'device': device,
        'model': model,
        'physics': physics,
        'geometry': geometry,
        'trainer': trainer,
        'visualizer': visualizer,
    }


def run_simulation(
    reynolds_number: float,
    inlet_velocity: float,
    radius: float,
    length: float,
    epochs: int,
    output_dir: str,
    run_id: Optional[str] = None,
    batch_interior: Optional[int] = None,
    batch_boundary: Optional[int] = None,
    lbfgs_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    components = build_simulation_components(
        reynolds_number=reynolds_number,
        inlet_velocity=inlet_velocity,
        radius=radius,
        length=length,
        run_id=run_id,
        log_dir=str(Path(output_dir) / 'logs'),
    )
    trainer: PINNTrainer = components['trainer']

    adam_epochs = int(epochs)
    lbfgs_epochs = int(lbfgs_epochs if lbfgs_epochs is not None else os.environ.get('LBFGS_EPOCHS', '100'))
    batch_interior = int(batch_interior if batch_interior is not None else os.environ.get('BATCH_INTERIOR', '1000'))
    batch_boundary = int(batch_boundary if batch_boundary is not None else os.environ.get('BATCH_BOUNDARY', '200'))

    logger.info(f"Starting simulation. Reynolds Number: {reynolds_number}")
    history = trainer.train(
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        batch_size_interior=batch_interior,
        batch_size_boundary=batch_boundary,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_state_path = output_path / f'pipe_flow_model_{run_id or "latest"}.pth'
    torch.save(components['model'].state_dict(), model_state_path)

    image_path = str(output_path / 'pipe_flow_result.png')
    components['visualizer'].plot_pipe_slice(length=float(length), radius=float(radius), save_path=image_path)

    return {
        'image_path': image_path,
        'history': history,
        'model_state_path': str(model_state_path),
    }


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
    REYNOLDS_NUMBER = float(os.environ.get("RE", "100.0"))
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
    outlet_suction_strength = float(os.environ.get('OUTLET_SUCTION_STRENGTH', '5.0'))
    outlet_suction_width = float(os.environ.get('OUTLET_SUCTION_WIDTH', '0.05'))
    geometry = PipeGeometrySampler(radius=radius, length=length)
    physics = NavierStokes3DPhysics(
        Re=REYNOLDS_NUMBER,
        spatial_weight_start=spatial_weight_start,
        spatial_weight_slope=spatial_weight_slope,
        pipe_length=length,
        outlet_suction_strength=outlet_suction_strength,
        outlet_suction_width=outlet_suction_width,
    )

    # 3. Setup Trainer
    log_dir = os.environ.get("LOGDIR", "logs")
    trainer = PINNTrainer(
        model=model,
        physics_engine=physics,
        geometry_sampler=geometry,
        device=device,
        lr=float(os.environ.get('LR', '1e-3')),
        lambda_bc=LAMBDA_BC,
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