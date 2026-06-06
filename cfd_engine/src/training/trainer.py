import os
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class PINNTrainer:
    """
    Endüstriyel Sınıf 3D PINN Eğitim Motoru.
    Aşama 1: Adam (Hızlı Keşif)
    Aşama 2: L-BFGS (Hassas Fiziksel Optimizasyon)
    """
    def __init__(
        self,
        model: nn.Module,
        physics_engine,
        geometry_sampler,
        device: torch.device,
        lr: float = 1e-3,
        lambda_bc: float = 20.0,
        log_dir: Optional[str] = "logs",
        grad_clip: Optional[float] = 1.0,
        ntk_check_interval: int = 1000,
        ntk_reg_weight: float = 1e-4,
        lambda_pin: float = 1.0,
        lambda_pos: float = 10.0,
        lambda_target_vel: float = 1000.0,
        lambda_inlet: float = 100.0,
        inlet_velocity: float = 1.0,
        pump_force_max: float = 0.1,
        pump_ramp_epochs: int = 200,
        run_id: Optional[str] = None,
    ):
        self.model = model
        self.physics = physics_engine
        self.geometry = geometry_sampler
        self.device = device
        self.lambda_bc = lambda_bc
        self.grad_clip = grad_clip
        self.ntk_check_interval = ntk_check_interval

        self.adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir) if log_dir is not None else None
        self.inlet_velocity = inlet_velocity
        self.ntk_reg_weight = ntk_reg_weight
        self.lambda_pin = lambda_pin
        self.lambda_pos = lambda_pos
        self.lambda_target_vel = lambda_target_vel
        self.lambda_inlet = lambda_inlet
        self.pump_force_max = pump_force_max
        self.pump_ramp_epochs = pump_ramp_epochs
        self.run_id = run_id if run_id is not None else ""

    def _compute_centerline_target_loss(self, model: nn.Module, num_points: int) -> torch.Tensor:
        L = getattr(self.geometry, 'length', 1.0)
        x_center = torch.rand((num_points, 1), device=self.device) * float(L)
        y_center = torch.zeros_like(x_center)
        z_center = torch.zeros_like(x_center)
        centerline_coords = torch.cat([x_center, y_center, z_center], dim=1)
        preds_centerline = model(centerline_coords)
        u_center = preds_centerline[:, 0:1]
        loss_target = torch.mean(torch.relu(1.0 - u_center) ** 2)
        return loss_target

    def train(self, adam_epochs: int, lbfgs_epochs: int, batch_size_interior: int = 2000, batch_size_boundary: int = 500) -> Dict[str, list]:
        self.model.train()
        history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'ntk': []}
        
        # ==========================================
        # AŞAMA 1: ADAM OPTIMIZER (Buldozer)
        # ==========================================
        logger.info(f"--- STAGE 1: Adam Optimization ({adam_epochs} Epochs) ---")
        for epoch in range(1, adam_epochs + 1):
            self.adam_optimizer.zero_grad()

            # Curriculum: ramp the artificial downstream pressure gradient (pump) slowly
            ramp_frac = min(1.0, epoch / max(1, self.pump_ramp_epochs))
            self.physics.artificial_pressure_gradient = float(self.pump_force_max * ramp_frac)

            interior_pts = self.geometry.sample_interior(batch_size_interior, self.device)
            wall_pts = self.geometry.sample_walls(batch_size_boundary, self.device)
            inlet_pts = self.geometry.sample_inlet(batch_size_boundary, self.device)

            loss_cont, loss_mom = self.physics.compute_pde_loss(self.model, interior_pts)
            
            preds_wall = self.model(wall_pts)
            loss_wall = torch.mean(preds_wall[:, 0:3]**2)

            preds_inlet = self.model(inlet_pts)
            y_in, z_in = inlet_pts[:, 1:2], inlet_pts[:, 2:3]
            r_in = torch.sqrt(y_in**2 + z_in**2)
            R = getattr(self.geometry, 'radius', 1.0)
            u_profile = self.inlet_velocity * torch.clamp(1.0 - (r_in / R) ** 2, min=0.0)
            loss_inlet = torch.mean((preds_inlet[:, 0:1] - u_profile) ** 2 + preds_inlet[:, 1:2] ** 2 + preds_inlet[:, 2:3] ** 2)
            loss_inlet = self.lambda_inlet * loss_inlet

            # Pressure pinning at outlet (x = length)
            L = getattr(self.geometry, 'length', 1.0)
            # create outlet points: x = L, random radial sampling
            x_out = torch.full((batch_size_boundary, 1), float(L), device=self.device)
            r = torch.sqrt(torch.rand((batch_size_boundary, 1), device=self.device)) * getattr(self.geometry, 'radius', 1.0)
            theta = torch.rand((batch_size_boundary, 1), device=self.device) * 2.0 * 3.141592653589793
            y_out = r * torch.cos(theta)
            z_out = r * torch.sin(theta)
            outlet_pts = torch.cat([x_out, y_out, z_out], dim=1).requires_grad_(False)

            loss_pin = self.physics.compute_pressure_pinning(self.model, outlet_pts, p_ref=0.0)
            loss_pos = self.physics.compute_positivity_loss(self.model, interior_pts)
            loss_target = self._compute_centerline_target_loss(self.model, batch_size_boundary)

            pde_loss = (loss_cont + loss_mom)
            bc_loss = (loss_wall + loss_inlet + self.lambda_pin * loss_pin + self.lambda_pos * loss_pos)

            # Dynamic weight balancing via gradient norms — more aggressive: amplify PDE when BC dominates
            params = [p for p in self.model.parameters() if p.requires_grad]
            eps = 1e-12
            try:
                grads_pde = torch.autograd.grad(pde_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
                norm_pde = torch.sqrt(sum([(g.detach() if g is None else g).norm()**2 for g in grads_pde if g is not None]) + eps)
                grads_bc = torch.autograd.grad(bc_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
                norm_bc = torch.sqrt(sum([(g.detach() if g is None else g).norm()**2 for g in grads_bc if g is not None]) + eps)
            except Exception:
                # fallback if autograd.grad fails for any reason
                norm_pde = torch.tensor(1.0, device=self.device)
                norm_bc = torch.tensor(1.0, device=self.device)

            # Compute PDE amplification weight: if BC gradients >> PDE gradients, increase PDE weight
            weight_pde_val = (norm_bc.detach() / (norm_pde.detach() + eps)).clamp(0.1, 100.0)
            # Be slightly aggressive (exponent) to favor PDE when needed
            weight_pde = weight_pde_val ** 1.5

            x_mean = torch.mean(interior_pts[:, 0:1])
            L = getattr(self.geometry, 'length', 1.0)
            propagation_factor = 1.0 + 3.0 * torch.sigmoid((x_mean / float(L) - 0.5) * 8.0)
            weight_pde = weight_pde * propagation_factor

            total_loss = weight_pde * pde_loss + self.lambda_bc * bc_loss + self.lambda_target_vel * loss_target

            # NTK regularizer occasionally (cheap small sample)
            ntk_reg = 0.0
            if epoch % self.ntk_check_interval == 0:
                try:
                    sample_pts = self.geometry.sample_interior(min(64, batch_size_interior), self.device)
                    K = self.physics.compute_empirical_ntk(self.model, sample_pts, output_index=0, max_points=32)
                    eigenvalues = torch.linalg.eigvalsh(K)
                    small = torch.clamp(eigenvalues[eigenvalues > 0.0], min=1e-12)
                    lambda_min = small.min() if small.numel() > 0 else torch.tensor(1e-12, device=self.device)
                    ntk_reg = float(self.ntk_reg_weight) / (lambda_min + 1e-12)
                except Exception as e:
                    logger.debug(f"NTK regularizer skipped: {e}")

            if isinstance(ntk_reg, float):
                total_loss = total_loss + ntk_reg

            # Logging pump force and dynamic weight occasionally
            if epoch % 100 == 0:
                logger.debug(f"Epoch {epoch}: pump_force={self.physics.artificial_pressure_gradient:.5e}, weight_pde={weight_pde:.3f}")

            total_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.adam_optimizer.step()

            if epoch % 50 == 0 or epoch == 1:
                logger.info(f"Adam Epoch: {epoch:04d}/{adam_epochs} | Loss: {total_loss.item():.4e}")
            if self.writer is not None:
                self.writer.add_scalar('train/total_loss', total_loss.item(), epoch)
                self.writer.add_scalar('train/pde_loss', pde_loss.item(), epoch)
                self.writer.add_scalar('train/bc_loss', bc_loss.item(), epoch)

            # --- AUTO-SAVE BLOCK ---
            if epoch % 500 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                suffix = f"_{self.run_id}" if self.run_id else ""
                save_path = f"checkpoints/auto_ckpt_epoch_{epoch}{suffix}.pth"
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Checkpoint saved at epoch {epoch}: {save_path}")

            # NTK diagnostics (expensive) - log occasionally
            if epoch % self.ntk_check_interval == 0:
                try:
                    sample_pts = self.geometry.sample_interior(min(128, batch_size_interior), self.device)
                    K = self.physics.compute_empirical_ntk(self.model, sample_pts, output_index=0, max_points=64)
                    metrics = self.physics.compute_ntk_spectral_metrics(K)
                    logger.info(f"NTK @ epoch {epoch}: cond={metrics['condition_number']:.3e}, slope={metrics['spectral_decay_slope']:.3f}")
                    if self.writer is not None:
                        self.writer.add_scalar('ntk/condition_number', metrics['condition_number'], epoch)
                        self.writer.add_scalar('ntk/spectral_decay_slope', metrics['spectral_decay_slope'], epoch)
                        history['ntk'].append(metrics)
                except Exception as e:
                    logger.warning(f"NTK diagnostics failed at epoch {epoch}: {e}")

        # ==========================================
        # AŞAMA 2: L-BFGS OPTIMIZER (Neşter)
        # ==========================================
        logger.info(f"--- STAGE 2: L-BFGS Optimization (Max {lbfgs_epochs} Iterations) ---")
        
        lbfgs_optimizer = torch.optim.LBFGS(
            self.model.parameters(), 
            max_iter=lbfgs_epochs, 
            tolerance_grad=1e-5, 
            tolerance_change=1e-9, 
            history_size=50
        )

        # Noktaları bir kez üretiyoruz
        interior_pts_base = self.geometry.sample_interior(batch_size_interior, self.device)
        wall_pts_base = self.geometry.sample_walls(batch_size_boundary, self.device)
        inlet_pts_base = self.geometry.sample_inlet(batch_size_boundary, self.device)

        def closure():
            # PyTorch L-BFGS PINN Çözümü: Her çağrıda grafiği tazelemek zorundayız.
            with torch.enable_grad():
                lbfgs_optimizer.zero_grad()
                
                # Tensorleri kopyala (clone), eski grafikten kopar (detach) ve türevi aç (requires_grad)
                curr_interior = interior_pts_base.clone().detach().requires_grad_(True)
                curr_wall = wall_pts_base.clone().detach().requires_grad_(True)
                curr_inlet = inlet_pts_base.clone().detach().requires_grad_(True)

                loss_cont, loss_mom = self.physics.compute_pde_loss(self.model, curr_interior)
                
                preds_wall = self.model(curr_wall)
                loss_wall = torch.mean(preds_wall[:, 0:3]**2)

                preds_inlet = self.model(curr_inlet)
                y_in, z_in = curr_inlet[:, 1:2], curr_inlet[:, 2:3]
                r_in = torch.sqrt(y_in**2 + z_in**2)
                R = getattr(self.geometry, 'radius', 1.0)
                u_profile = self.inlet_velocity * torch.clamp(1.0 - (r_in / R) ** 2, min=0.0)
                loss_inlet = torch.mean((preds_inlet[:, 0:1] - u_profile) ** 2 + preds_inlet[:, 1:2] ** 2 + preds_inlet[:, 2:3] ** 2)
                loss_inlet = self.lambda_inlet * loss_inlet
                # pressure pinning (outlet)
                L = getattr(self.geometry, 'length', 1.0)
                x_out = torch.full((curr_inlet.shape[0], 1), float(L), device=self.device)
                r = torch.sqrt(torch.rand((curr_inlet.shape[0], 1), device=self.device)) * getattr(self.geometry, 'radius', 1.0)
                theta = torch.rand((curr_inlet.shape[0], 1), device=self.device) * 2.0 * 3.141592653589793
                y_out = r * torch.cos(theta)
                z_out = r * torch.sin(theta)
                outlet_pts = torch.cat([x_out, y_out, z_out], dim=1).requires_grad_(False)

                loss_pin = self.physics.compute_pressure_pinning(self.model, outlet_pts, p_ref=0.0)
                loss_pos = self.physics.compute_positivity_loss(self.model, curr_interior)
                loss_target = self._compute_centerline_target_loss(self.model, curr_inlet.shape[0])

                pde_loss = (loss_cont + loss_mom)
                bc_loss = (loss_wall + loss_inlet + self.lambda_pin * loss_pin + self.lambda_pos * loss_pos)

                params = [p for p in self.model.parameters() if p.requires_grad]
                eps = 1e-12
                try:
                    grads_pde = torch.autograd.grad(pde_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
                    norm_pde = torch.sqrt(sum([(g.detach() if g is None else g).norm()**2 for g in grads_pde if g is not None]) + eps)
                    grads_bc = torch.autograd.grad(bc_loss, params, retain_graph=True, create_graph=True, allow_unused=True)
                    norm_bc = torch.sqrt(sum([(g.detach() if g is None else g).norm()**2 for g in grads_bc if g is not None]) + eps)
                except Exception:
                    norm_pde = torch.tensor(1.0, device=self.device)
                    norm_bc = torch.tensor(1.0, device=self.device)

                weight_pde_val = (norm_bc.detach() / (norm_pde.detach() + eps)).clamp(0.1, 100.0)
                weight_pde = weight_pde_val ** 1.5

                x_mean = torch.mean(curr_interior[:, 0:1])
                L = getattr(self.geometry, 'length', 1.0)
                propagation_factor = 1.0 + 3.0 * torch.sigmoid((x_mean / float(L) - 0.5) * 8.0)
                weight_pde = weight_pde * propagation_factor

                total_loss = weight_pde * pde_loss + self.lambda_bc * bc_loss + self.lambda_target_vel * loss_target

                # small NTK regularizer during L-BFGS closure
                try:
                    sample_pts = self.geometry.sample_interior(min(64, curr_interior.shape[0]), self.device)
                    K = self.physics.compute_empirical_ntk(self.model, sample_pts, output_index=0, max_points=32)
                    eigenvalues = torch.linalg.eigvalsh(K)
                    small = torch.clamp(eigenvalues[eigenvalues > 0.0], min=1e-12)
                    lambda_min = small.min() if small.numel() > 0 else torch.tensor(1e-12, device=self.device)
                    total_loss = total_loss + float(self.ntk_reg_weight) / (lambda_min + 1e-12)
                except Exception:
                    pass

                total_loss.backward()

                logger.info(f"L-BFGS Step Loss: {total_loss.item():.4e}")
                return total_loss

        lbfgs_optimizer.step(closure)
        logger.info("Training completed successfully.")
        return history