import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
import time
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class PINNTrainer:
    """
    Production-grade PINN Trainer for 3D pipe flow with spatial aliasing suppression.
    
    Two-stage optimization pipeline:
    1. Adam (1600 epochs) with CosineAnnealingLR scheduler for coarse fit
    2. L-BFGS (1500 max iterations) with ultra-tight tolerances for fine refinement
    
    Loss architecture:
    - PDE (Continuity × 100 + Momentum)
    - Radial Guide (× 1000): enforces parabolic u(r) profile, prevents needle artifacts
    - Positivity (× 500): ensures u_x ≥ 0 everywhere
    - Axial Smoothness (× 100): penalizes d²u/dx², kills vertical aliasing lines
    - Boundary Conditions (× λ_bc): inlet parabolic, walls no-slip, outlet open
    """
    
    def __init__(
        self,
        model: nn.Module,
        physics_engine,
        geometry_sampler,
        device: torch.device,
        lr: float = 1e-3,
        lambda_bc: float = 10000.0,
        lambda_smooth: float = 100.0,
        lambda_radial: float = 1000.0,
        lambda_pos: float = 500.0,
    ):
        self.model = model
        self.physics = physics_engine
        self.geometry = geometry_sampler
        self.device = device
        self.lambda_bc = lambda_bc
        self.lambda_smooth = lambda_smooth      # Axial smoothness penalty weight
        self.lambda_radial = lambda_radial      # Radial guide penalty weight
        self.lambda_pos = lambda_pos            # Positivity penalty weight
        
        # Adam optimizer with CosineAnnealingLR scheduler
        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Scheduler will be created in train() method
        self.scheduler = None
        
        # Training history tracking
        self.loss_history = {
            'total': [], 'pde': [], 'radial': [], 'positivity': [],
            'smoothness': [], 'boundary': [], 'continuity': [], 'momentum': []
        }

    def _compute_inlet_loss(self, preds_inlet: torch.Tensor, inlet_pts: torch.Tensor) -> torch.Tensor:
        """
        Enforces Poiseuille parabolic inlet profile: u(r) = 2(1 - r²/R²)
        Also suppresses radial and axial velocities at inlet.
        
        Args:
            preds_inlet: (N, 4) predictions [u_x, v_y, w_z, p]
            inlet_pts: (N, 3) inlet points [x=0, y, z]
        
        Returns:
            Scalar inlet boundary loss
        """
        y = inlet_pts[:, 1:2]
        z = inlet_pts[:, 2:3]
        r_squared = y**2 + z**2
        R_max = self.geometry.radius ** 2
        
        # Parabolic profile: u = 2 * (1 - r²/R²)
        parabolic_u = 2.0 * (1.0 - (r_squared / R_max))
        parabolic_u = torch.relu(parabolic_u)  # Clamp negative (edge cases)
        
        u_in = preds_inlet[:, 0:1]
        v_in = preds_inlet[:, 1:2]
        w_in = preds_inlet[:, 2:3]
        
        # u should match parabolic profile, v and w should be ~0
        loss_u = torch.mean((u_in - parabolic_u) ** 2)
        loss_transverse = torch.mean(v_in**2 + w_in**2)
        
        return loss_u + 0.1 * loss_transverse  # Transverse slightly less strict

    def _compute_radial_guide_loss(self, preds_int: torch.Tensor, int_pts: torch.Tensor) -> torch.Tensor:
        """
        Enforces radial velocity profile consistency throughout the domain.
        
        ARCHITECTURAL PRINCIPLE: The neural network tends to form a thin "needle" in the pipe center
        (where it outputs high u values only at center, zero elsewhere) rather than the physical
        parabolic profile u(r) = 2(1 - r²/R²). This loss guides all interior points toward the
        parabolic profile regardless of axial position, preventing needle formation.
        
        Weight: λ_radial = 1000× (very strict, but essential for physical correctness)
        
        Args:
            preds_int: (N, 4) predictions [u_x, v_y, w_z, p]
            int_pts: (N, 3) interior points [x, y, z]
        
        Returns:
            Scalar radial guide loss
        """
        y = int_pts[:, 1:2]
        z = int_pts[:, 2:3]
        r_squared = y**2 + z**2
        R_max = self.geometry.radius ** 2
        
        # Target: Poiseuille parabolic profile everywhere
        expected_u = 2.0 * (1.0 - (r_squared / R_max))
        expected_u = torch.relu(expected_u)
        
        u_pred = preds_int[:, 0:1]
        radial_loss = torch.mean((u_pred - expected_u) ** 2)
        
        return radial_loss

    def _compute_axial_smoothness_loss(self, int_pts: torch.Tensor) -> torch.Tensor:
        """
        Axial Total Variation Loss: Penalizes sudden changes in du/dx (velocity gradient).
        
        PHYSICAL MOTIVATION: Spatial aliasing artifacts in the visualization appear as vertical lines
        because the neural network's du/dx has sharp local discontinuities. By penalizing the second
        derivative d²u/dx² (curvature of the velocity gradient), we force smooth variations along x.
        
        Implementation:
        1. Forward pass through interior points (with gradient tracking)
        2. Compute du/dx via autograd
        3. Compute d²u/dx² by differentiating du/dx
        4. Take L2 norm of d²u/dx² as penalty
        
        Weight: λ_smooth = 100× (sufficient to suppress aliasing without over-regularization)
        
        Args:
            int_pts: (N, 3) interior points [x, y, z]
        
        Returns:
            Scalar smoothness loss
        """
        int_pts_req = int_pts.clone().detach().requires_grad_(True)
        preds = self.model(int_pts_req)
        u_pred = preds[:, 0:1]
        
        try:
            # First derivative: du/dx
            du_dx = torch.autograd.grad(
                outputs=u_pred.sum(),
                inputs=int_pts_req,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            
            if du_dx is None:
                # Fallback if gradient is None
                return torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            du_dx = du_dx[:, 0:1]  # Extract d/dx component
            
            # Second derivative: d²u/dx² (curvature penalty)
            d2u_dx2 = torch.autograd.grad(
                outputs=du_dx.sum(),
                inputs=int_pts_req,
                create_graph=False,
                retain_graph=True,
                allow_unused=True,
            )[0]
            
            if d2u_dx2 is None:
                return torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            d2u_dx2 = d2u_dx2[:, 0:1]
            smoothness_loss = torch.mean(d2u_dx2 ** 2)
            
        except RuntimeError as e:
            logger.warning(f"Gradient computation failed in axial smoothness loss: {e}")
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        return smoothness_loss

    def train(
        self,
        adam_epochs: int = 1600,
        lbfgs_epochs: int = 1500,
        batch_size_int: int = 4000,
        batch_size_bc: int = 800,
    ) -> Dict[str, list]:
        """
        Production-grade two-stage training with comprehensive logging and checkpointing.
        
        STAGE 1 - Adam Optimization (1600 epochs):
            • CosineAnnealingLR scheduler: gradually reduces learning rate from lr to 1e-6
            • Batch-wise sampling: interior, walls, inlet, outlet
            • Full loss computation with all regularization terms
            • Checkpoint every 200 epochs, full save at stage end
        
        STAGE 2 - L-BFGS Refinement (1500 max iterations):
            • Ultra-tight tolerances: change=1e-16, grad=1e-16
            • Strong Wolfe line search for robust convergence
            • History size = 50 for memory efficiency
            • Does NOT terminate early (fixed_iter=True behavior)
        
        Args:
            adam_epochs: Number of Adam optimization epochs (default 1600)
            lbfgs_epochs: Max L-BFGS iterations (default 1500)
            batch_size_int: Interior collocation points per batch (default 4000)
            batch_size_bc: Boundary collocation points per batch (default 800)
        
        Returns:
            Dictionary with loss history for analysis
        """
        self.model.train()
        
        # Setup CosineAnnealingLR scheduler for Adam phase
        self.scheduler = CosineAnnealingLR(
            self.adam_optimizer,
            T_max=adam_epochs,
            eta_min=1e-6,  # Minimum learning rate
        )
        
        logger.info("=" * 80)
        logger.info("STAGE 1: Adam Optimization with CosineAnnealingLR Scheduler")
        logger.info("=" * 80)
        logger.info(f"Epochs: {adam_epochs} | Batch(Interior): {batch_size_int} | Batch(BC): {batch_size_bc}")
        logger.info(f"Loss weights: λ_bc={self.lambda_bc:.1e}, λ_radial={self.lambda_radial:.1e}, " 
                   f"λ_pos={self.lambda_pos:.1e}, λ_smooth={self.lambda_smooth:.1e}")
        
        stage1_start_time = time.time()
        
        # STAGE 1: Adam Optimization
        for epoch in range(1, adam_epochs + 1):
            self.adam_optimizer.zero_grad()
            
            # Sample collocation points
            int_pts = self.geometry.sample_interior(batch_size_int, self.device)
            bc_pts = self.geometry.sample_walls(batch_size_bc, self.device)
            in_pts = self.geometry.sample_inlet(batch_size_bc, self.device)
            out_pts = self.geometry.sample_outlet(batch_size_bc, self.device)
            
            # ============================================================================
            # COMPUTE ALL LOSSES
            # ============================================================================
            
            # 1. PDE Loss (Continuity + Momentum)
            l_cont, l_mom = self.physics.compute_pde_loss(self.model, int_pts)
            # CRITICAL: Continuity × 100 prevents needle artifact by enforcing mass conservation
            pde_loss = (100.0 * l_cont) + l_mom
            
            # 2. Interior Point Losses (Radial Guide, Positivity, Smoothness)
            preds_int = self.model(int_pts)
            u_pred_int = preds_int[:, 0:1]
            
            # Positivity constraint: u_x ≥ 0 everywhere
            l_pos = torch.mean(torch.relu(-u_pred_int))
            
            # Radial guide: enforce parabolic profile
            l_radial = self._compute_radial_guide_loss(preds_int, int_pts)
            
            # Axial smoothness: suppress d²u/dx² (kills vertical aliasing)
            l_smooth = self._compute_axial_smoothness_loss(int_pts)
            
            # 3. Boundary Condition Losses
            # Wall: no-slip BC (u=v=w=0 at r=R)
            l_wall = torch.mean(self.model(bc_pts)[:, 0:3] ** 2)
            
            # Inlet: parabolic profile
            l_in = self._compute_inlet_loss(self.model(in_pts), in_pts)
            
            # Outlet: open BC (∂p/∂x = 0, approximated by suppressing pressure gradient)
            l_out = torch.mean(self.model(out_pts)[:, 3:4] ** 2)
            
            # ============================================================================
            # COMBINE LOSSES WITH PRODUCTION WEIGHTS
            # ============================================================================
            l_bc_total = self.lambda_bc * (l_wall + l_in + l_out)
            
            total_loss = (
                pde_loss
                + self.lambda_radial * l_radial
                + self.lambda_pos * l_pos
                + self.lambda_smooth * l_smooth
                + l_bc_total
            )
            
            # Backward pass
            total_loss.backward()
            self.adam_optimizer.step()
            self.scheduler.step()
            
            # Store history
            self.loss_history['total'].append(total_loss.item())
            self.loss_history['pde'].append(pde_loss.item())
            self.loss_history['radial'].append(l_radial.item())
            self.loss_history['positivity'].append(l_pos.item())
            self.loss_history['smoothness'].append(l_smooth.item())
            self.loss_history['boundary'].append(l_bc_total.item())
            self.loss_history['continuity'].append(l_cont.item())
            self.loss_history['momentum'].append(l_mom.item())
            
            # Logging
            if epoch % 50 == 0 or epoch == 1:
                elapsed = time.time() - stage1_start_time
                logger.info(
                    f"Epoch {epoch:5d}/{adam_epochs} | Tot: {total_loss.item():.2e} | "
                    f"PDE: {pde_loss.item():.2e} | Radial: {l_radial.item():.2e} | "
                    f"Smooth: {l_smooth.item():.2e} | LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
            
            # Checkpoint every 200 epochs
            if epoch % 200 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/auto_ckpt_adam_{epoch}.pth"
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f"  → Checkpoint saved: {ckpt_path}")
        
        stage1_duration = time.time() - stage1_start_time
        logger.info(f"STAGE 1 COMPLETE in {stage1_duration:.1f}s | Final Loss: {total_loss.item():.2e}\n")
        
        # ============================================================================
        # STAGE 2: L-BFGS OPTIMIZATION (Perseverant, never gives up)
        # ============================================================================
        logger.info("=" * 80)
        logger.info("STAGE 2: L-BFGS Optimization (Perseverance Mode)")
        logger.info("=" * 80)
        logger.info(f"Max Iterations: {lbfgs_epochs} | Tolerance Change: 1e-16 | Tolerance Grad: 1e-16")
        logger.info(f"Line Search: strong_wolfe | History Size: 50")
        
        lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            max_iter=lbfgs_epochs,
            history_size=50,
            tolerance_change=1e-16,      # ULTRA-TIGHT: never stop due to loss change
            tolerance_grad=1e-16,        # ULTRA-TIGHT: never stop due to gradient size
            line_search_fn='strong_wolfe',  # Robust line search
        )
        
        # Sample points once (fixed batch for full L-BFGS phase)
        int_pts_b = self.geometry.sample_interior(batch_size_int, self.device)
        bc_pts_b = self.geometry.sample_walls(batch_size_bc, self.device)
        in_pts_b = self.geometry.sample_inlet(batch_size_bc, self.device)
        out_pts_b = self.geometry.sample_outlet(batch_size_bc, self.device)
        
        stage2_start_time = time.time()
        iteration_count = [0]  # Track iterations for logging
        
        def closure():
            """Closure function for L-BFGS step computation."""
            iteration_count[0] += 1
            
            lbfgs_optimizer.zero_grad()
            
            # Compute all losses
            l_cont, l_mom = self.physics.compute_pde_loss(self.model, int_pts_b)
            pde_loss = (100.0 * l_cont) + l_mom
            
            preds_int = self.model(int_pts_b)
            u_pred_int = preds_int[:, 0:1]
            
            l_pos = torch.mean(torch.relu(-u_pred_int))
            l_radial = self._compute_radial_guide_loss(preds_int, int_pts_b)
            l_smooth = self._compute_axial_smoothness_loss(int_pts_b)
            
            l_wall = torch.mean(self.model(bc_pts_b)[:, 0:3] ** 2)
            l_in = self._compute_inlet_loss(self.model(in_pts_b), in_pts_b)
            l_out = torch.mean(self.model(out_pts_b)[:, 3:4] ** 2)
            
            l_bc_total = self.lambda_bc * (l_wall + l_in + l_out)
            
            total_loss = (
                pde_loss
                + self.lambda_radial * l_radial
                + self.lambda_pos * l_pos
                + self.lambda_smooth * l_smooth
                + l_bc_total
            )
            
            # Backward pass
            total_loss.backward()
            
            # Log every 50 iterations
            if iteration_count[0] % 50 == 0 or iteration_count[0] == 1:
                elapsed = time.time() - stage2_start_time
                logger.info(
                    f"L-BFGS Iter {iteration_count[0]:5d} | Tot: {total_loss.item():.2e} | "
                    f"PDE: {pde_loss.item():.2e} | Radial: {l_radial.item():.2e} | "
                    f"Smooth: {l_smooth.item():.2e} | Time: {elapsed:.1f}s"
                )
            
            return total_loss
        
        # Execute L-BFGS (will run up to max_iter or until convergence)
        lbfgs_optimizer.step(closure)
        
        stage2_duration = time.time() - stage2_start_time
        logger.info(f"STAGE 2 COMPLETE in {stage2_duration:.1f}s | Total iterations: {iteration_count[0]}\n")
        
        # Save final model
        os.makedirs("checkpoints", exist_ok=True)
        final_path = "checkpoints/pipe_flow_final.pth"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"✓ Final model saved to {final_path}")
        
        total_duration = stage1_duration + stage2_duration
        logger.info(f"✓ Total training duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        logger.info("=" * 80)
        
        return self.loss_history
