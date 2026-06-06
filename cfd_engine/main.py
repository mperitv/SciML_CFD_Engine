import torch
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('cfd_simulation.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("SciML_CFD_Engine")

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics
from src.geometry.sdf_sampler import PipeGeometrySampler
from src.post_processing.visualizer import CFDVisualizer
from src.training.trainer import PINNTrainer


# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================

def build_simulation_components(
    reynolds_number: float,
    radius: float,
    length: float,
) -> Dict[str, Any]:
    """
    Initialize all 3D PINN CFD engine components with production configuration.
    
    Args:
        reynolds_number: Reynolds number for simulation
        radius: Pipe radius in physical units
        length: Pipe length in physical units
    
    Returns:
        Dictionary containing initialized: device, model, physics, geometry, trainer, visualizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing SciML CFD Engine on {device.type.upper()}")
    
    # ========================================================================
    # RANDOMNESS CONTROL (Reproducibility)
    # ========================================================================
    seed = int(os.environ.get('SEED', '42'))
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
    
    # ========================================================================
    # GEOMETRY
    # ========================================================================
    geometry = PipeGeometrySampler(radius=float(radius), length=float(length))
    logger.info(f"Geometry: Pipe radius={radius}, length={length}")
    
    # ========================================================================
    # PHYSICS ENGINE
    # ========================================================================
    spatial_weight_start    = float(os.environ.get('SPATIAL_WEIGHT_START',    '1.0'))
    spatial_weight_slope    = float(os.environ.get('SPATIAL_WEIGHT_SLOPE',    '1.0'))
    outlet_suction_strength = float(os.environ.get('OUTLET_SUCTION_STRENGTH', '5.0'))
    outlet_suction_width    = float(os.environ.get('OUTLET_SUCTION_WIDTH',    '0.05'))

    # Hagen-Poiseuille analitik itme kuvveti: f = 8*nu/R^2
    # Re=100, R=0.5 → f = 8/(100*0.25) = 0.32  |  Re=50, R=0.5 → f = 0.64
    # Bu deger warm-up'ta ogretilen p_exact = f*(L-x) ile tutarlidir.
    nu_val        = 1.0 / float(reynolds_number)
    driving_force = float(os.environ.get(
        'DRIVING_FORCE',
        str(8.0 * nu_val / (float(radius) ** 2))
    ))

    physics = NavierStokes3DPhysics(
        Re=float(reynolds_number),
        spatial_weight_start=spatial_weight_start,
        spatial_weight_slope=spatial_weight_slope,
        pipe_length=float(length),
        outlet_suction_strength=outlet_suction_strength,
        outlet_suction_width=outlet_suction_width,
        driving_force=driving_force,
    )
    logger.info(
        f"Physics: Re={reynolds_number}, nu={nu_val:.4f}, "
        f"driving_force={driving_force:.4f} | spatial_weight_start={spatial_weight_start}"
    )
    
    # ========================================================================
    # NEURAL NETWORK MODEL
    # ========================================================================
    hidden_dim = int(os.environ.get('HIDDEN_DIM', '256'))
    num_layers = int(os.environ.get('NUM_LAYERS', '6'))
    
    model = PINN3DEngine(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        length=float(length),
        radius=float(radius)
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: PINN3DEngine with {hidden_dim}D embeddings, {num_layers} layers, {param_count:,} parameters")
    
    # ========================================================================
    # TRAINER WITH PRODUCTION WEIGHTS (Golden Ratio Tuning for Poiseuille)
    # ========================================================================
    lr        = float(os.environ.get('LR', '1e-3'))
    lambda_bc = float(os.environ.get('LAMBDA_BC', '100.0'))  # Pre-training sonrası sakin BC

    trainer = PINNTrainer(
        model=model,
        physics_engine=physics,
        geometry_sampler=geometry,
        device=device,
        lr=lr,
        lambda_bc=lambda_bc,
    )
    logger.info(
        f"Trainer initialized: lr={lr:.2e}, lambda_bc={lambda_bc:.1f} | "
        f"Physics weights: PDE=1x, radial=1x, pos=10x, smooth=1x"
    )
    
    # ========================================================================
    # VISUALIZER
    # ========================================================================
    visualizer = CFDVisualizer(model, device)
    logger.info("Visualizer initialized")
    
    return {
        'device': device,
        'model': model,
        'physics': physics,
        'geometry': geometry,
        'trainer': trainer,
        'visualizer': visualizer,
    }


# ============================================================================
# SIMULATION EXECUTION
# ============================================================================

def run_simulation(
    reynolds_number: float,
    radius: float,
    length: float,
    adam_epochs: int = 2000,
    lbfgs_epochs: int = 1000,
    batch_size_int: int = 4000,
    batch_size_bc: int = 800,
    warmup_epochs: int = 500,
    output_dir: str = "output",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute complete 3D pipe flow PINN simulation (API interface).

    Three-stage training:
    1. Warm-Up: 500 epochs — analitik Poiseuille fit (PDE yok)
    2. Physics Adam: 1500 epochs — N-S + BC, sakin katsayılar
    3. L-BFGS: 1000 iterations — 1e-16 toleranslı ince ayar

    Args:
        reynolds_number: Flow Reynolds number
        radius: Pipe radius
        length: Pipe length
        adam_epochs: Toplam Adam epoch sayısı (warm-up dahil, default 2000)
        lbfgs_epochs: L-BFGS max iterasyon sayısı (default 1000)
        batch_size_int: Interior batch size (default 4000)
        batch_size_bc: Boundary batch size (default 800)
        warmup_epochs: Analytical pre-training epochs (default 500)
        output_dir: Directory for results
        run_id: Optional run identifier

    Returns:
        Dictionary with image_path, history, model_state_path
    """
    logger.info("=" * 80)
    logger.info("STARTING FULL SIMULATION")
    logger.info("=" * 80)
    
    # Build all components
    components = build_simulation_components(
        reynolds_number=reynolds_number,
        radius=radius,
        length=length,
    )
    trainer: PINNTrainer = components['trainer']
    
    # Execute two-stage training
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING CONFIG: Adam={adam_epochs}e, L-BFGS={lbfgs_epochs}i, Batch(int/bc)={batch_size_int}/{batch_size_bc}")
    logger.info("=" * 80)
    
    history = trainer.train(
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        batch_size_int=batch_size_int,
        batch_size_bc=batch_size_bc,
        warmup_epochs=warmup_epochs,
    )

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    run_suffix = f"_{run_id}" if run_id else ""
    model_state_path = output_path / f'pipe_flow_model{run_suffix}.pth'
    torch.save(components['model'].state_dict(), model_state_path)
    logger.info(f"Model saved: {model_state_path}")
    
    # Generate visualization
    image_path = str(output_path / f'pipe_flow_result{run_suffix}.png')
    components['visualizer'].plot_pipe_slice(
        length=float(length),
        radius=float(radius),
        save_path=image_path
    )
    logger.info(f"Visualization saved: {image_path}")
    
    logger.info("=" * 80)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 80)
    
    return {
        'image_path': image_path,
        'history': history,
        'model_state_path': str(model_state_path),
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """
    Production CLI entry point for 3D PINN pipe flow simulation.
    
    All parameters configurable via environment variables:
    
    GEOMETRY:
        RE=100.0              Reynolds number (laminar Poiseuille, p_coeff=0.32)
        RADIUS=0.5            Pipe radius
        LENGTH=3.0            Pipe length
    
    TRAINING (Stage 1 - Adam):
        ADAM_EPOCHS=2000      Number of Adam optimization epochs
        LR=1e-3               Learning rate
    
    TRAINING (Stage 2 - L-BFGS):
        LBFGS_EPOCHS=1000     Max L-BFGS iterations
    
    BATCHING:
        BATCH_INTERIOR=4000   Interior collocation points/batch
        BATCH_BOUNDARY=800    Boundary collocation points/batch
    
    TRAINING:
        WARMUP_EPOCHS=500     Analytical pre-training (no PDE)
        ADAM_EPOCHS=2000      Total Adam epochs (warmup + physics)
        LBFGS_EPOCHS=1000     L-BFGS max iterations

    LOSS WEIGHTS (Hardcoded — Pre-training sonrasi sakin degerler):
        PDE = l_cont + l_mom  (1x — warm-up sayesinde yeterli)
        radial=1.0  pos=10.0  smooth=1.0
        LAMBDA_BC=100.0       Wall BC (pre-training ile zaten ogrenildi)
    
    PHYSICS:
        SPATIAL_WEIGHT_START=1.0      Initial spatial weighting
        SPATIAL_WEIGHT_SLOPE=1.0      Spatial weight slope
        OUTLET_SUCTION_STRENGTH=5.0   Outlet BC strength
        OUTLET_SUCTION_WIDTH=0.05     Outlet BC width
    
    MODEL:
        HIDDEN_DIM=256        Fourier embedding dimension
        NUM_LAYERS=6          Number of MLP layers
    
    OTHER:
        SEED=42               Random seed for reproducibility
        LOGDIR=logs           Log directory
        RUN_ID=                Optional run identifier suffix
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("\n" + "=" * 80)
    logger.info("SciML CFD ENGINE - PRODUCTION PIPE FLOW SOLVER")
    logger.info("=" * 80)
    logger.info(f"Device: {device.type.upper()}")
    
    # ========================================================================
    # LOAD CONFIGURATION FROM ENVIRONMENT
    # ========================================================================
    
    # Geometry
    reynolds_number = float(os.environ.get('RE', '100.0'))
    radius = float(os.environ.get('RADIUS', '0.5'))
    length = float(os.environ.get('LENGTH', '3.0'))

    # Training schedule
    adam_epochs   = int(os.environ.get('ADAM_EPOCHS',   '2000'))
    lbfgs_epochs  = int(os.environ.get('LBFGS_EPOCHS',  '1000'))
    warmup_epochs = int(os.environ.get('WARMUP_EPOCHS', '500'))

    # Batching
    batch_size_int = int(os.environ.get('BATCH_INTERIOR', '4000'))
    batch_size_bc  = int(os.environ.get('BATCH_BOUNDARY', '800'))

    logger.info(f"Geometry: Re={reynolds_number}, R={radius}, L={length}")
    logger.info(
        f"Training: Warm-Up={warmup_epochs}e | "
        f"Physics Adam={adam_epochs - warmup_epochs}e | "
        f"L-BFGS={lbfgs_epochs}i"
    )
    logger.info(f"Batching: Interior={batch_size_int}, BC={batch_size_bc}")
    logger.info(
        f"Loss weights: lambda_bc=100.0 | PDE=1x(cont+mom) | "
        f"radial=1x | pos=10x | smooth=1x"
    )
    
    # ========================================================================
    # BUILD COMPONENTS
    # ========================================================================
    components = build_simulation_components(
        reynolds_number=reynolds_number,
        radius=radius,
        length=length,
    )
    trainer: PINNTrainer = components['trainer']
    model = components['model']
    
    # ========================================================================
    # EXECUTE TRAINING
    # ========================================================================
    logger.info("\n" + "█" * 80)
    logger.info("STARTING TWO-STAGE OPTIMIZATION")
    logger.info("█" * 80)
    
    history = trainer.train(
        adam_epochs=adam_epochs,
        lbfgs_epochs=lbfgs_epochs,
        batch_size_int=batch_size_int,
        batch_size_bc=batch_size_bc,
        warmup_epochs=warmup_epochs,
    )

    # ========================================================================
    # SAVE AND VISUALIZE RESULTS
    # ========================================================================
    os.makedirs("checkpoints", exist_ok=True)
    run_id = os.environ.get('RUN_ID', '')
    suffix = f"_{run_id}" if run_id else ""
    
    model_path = f"checkpoints/pipe_flow_model{suffix}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")
    
    try:
        image_path = f"output/pipe_flow_visualization{suffix}.png"
        os.makedirs("output", exist_ok=True)
        components['visualizer'].plot_pipe_slice(
            length=length,
            radius=radius,
            save_path=image_path
        )
        logger.info(f"✓ Visualization saved to: {image_path}")
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()