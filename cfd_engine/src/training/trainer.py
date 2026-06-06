import torch
import torch.nn as nn
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

class PINNTrainer:
    def __init__(self, model: nn.Module, physics_engine, geometry_sampler, device: torch.device, lr: float = 1e-3, lambda_bc: float = 10000.0):
        self.model = model
        self.physics = physics_engine
        self.geometry = geometry_sampler
        self.device = device
        self.lambda_bc = lambda_bc
        self.adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_inlet_loss(self, preds_inlet, inlet_pts):
        y, z = inlet_pts[:, 1:2], inlet_pts[:, 2:3]
        r_squared = y**2 + z**2
        R_max = self.geometry.radius**2
        parabolic_u = 2.0 * (1.0 - (r_squared / R_max))
        parabolic_u = torch.relu(parabolic_u) # Negatif hedef olmasını engeller
        
        u_in, v_in, w_in = preds_inlet[:, 0:1], preds_inlet[:, 1:2], preds_inlet[:, 2:3]
        return torch.mean((u_in - parabolic_u)**2 + v_in**2 + w_in**2)

    def _compute_radial_guide_loss(self, preds_int, int_pts):
        """
        CTO MÜDAHALESİ: Akışı merkezden kenarlara zorla yayar.
        Yapay zekanın ince bir iplik (needle) çizmesini kesin olarak engeller.
        """
        y, z = int_pts[:, 1:2], int_pts[:, 2:3]
        r_squared = y**2 + z**2
        R_max = self.geometry.radius**2
        
        # İçerideki suyun da parabole benzemesi gerektiğini dayatıyoruz
        expected_u = 2.0 * (1.0 - (r_squared / R_max))
        expected_u = torch.relu(expected_u)
        
        u_pred = preds_int[:, 0:1]
        return torch.mean((u_pred - expected_u)**2)

    def train(
        self,
        adam_epochs: int,
        lbfgs_epochs: int,
        batch_size_int: int = 4000,
        batch_size_bc: int = 800,
        batch_size_interior: int | None = None,
        batch_size_boundary: int | None = None,
    ) -> Dict[str, list]:
        batch_size_int = batch_size_interior if batch_size_interior is not None else batch_size_int
        batch_size_bc = batch_size_boundary if batch_size_boundary is not None else batch_size_bc
        self.model.train()
        history = {'loss': []}
        
        logger.info(f"--- STAGE 1: Adam Optimization ({adam_epochs} Epochs) ---")
        for epoch in range(1, adam_epochs + 1):
            self.adam_optimizer.zero_grad()

            int_pts = self.geometry.sample_interior(batch_size_int, self.device)
            bc_pts = self.geometry.sample_walls(batch_size_bc, self.device)
            in_pts = self.geometry.sample_inlet(batch_size_bc, self.device)
            out_pts = self.geometry.sample_outlet(batch_size_bc, self.device)

            # 1. Fizik (PDE) Kayıpları
            l_cont, l_mom = self.physics.compute_pde_loss(self.model, int_pts)
            # Suyun sıkışmasını engellemek için Continuity'yi 100 ile çarpıyoruz!
            pde_loss = (100.0 * l_cont) + l_mom 
            
            # 2. Radyal Yayılım ve Pozitiflik
            preds_int = self.model(int_pts)
            l_pos = torch.mean(torch.relu(-preds_int[:, 0:1]))
            l_radial = self._compute_radial_guide_loss(preds_int, int_pts)

            # 3. Sınır Koşulları (BC)
            l_wall = torch.mean(self.model(bc_pts)[:, 0:3]**2)
            l_in = self._compute_inlet_loss(self.model(in_pts), in_pts)
            l_out = torch.mean(self.model(out_pts)[:, 3:4]**2)
            
            # TOTAL LOSS: Radyal kılavuzu 1000 çarpanıyla dayatıyoruz!
            total_loss = pde_loss + (1000.0 * l_radial) + (500.0 * l_pos) + self.lambda_bc * (l_wall + l_in + l_out)
            
            total_loss.backward()
            self.adam_optimizer.step()

            if epoch % 100 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:04d} | Tot: {total_loss.item():.2e} | Radial: {l_radial.item():.2e} | Cont: {l_cont.item():.2e}")
                if epoch % 500 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(self.model.state_dict(), f"checkpoints/auto_ckpt_{epoch}.pth")

        logger.info(f"--- STAGE 2: L-BFGS Optimization ---")
        lbfgs_optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=lbfgs_epochs, history_size=50)

        int_pts_b = self.geometry.sample_interior(batch_size_int, self.device)
        bc_pts_b = self.geometry.sample_walls(batch_size_bc, self.device)
        in_pts_b = self.geometry.sample_inlet(batch_size_bc, self.device)
        out_pts_b = self.geometry.sample_outlet(batch_size_bc, self.device)

        def closure():
            with torch.enable_grad():
                lbfgs_optimizer.zero_grad()
                
                c_int = int_pts_b.clone().detach().requires_grad_(True)
                c_bc = bc_pts_b.clone().detach().requires_grad_(True)
                c_in = in_pts_b.clone().detach().requires_grad_(True)
                c_out = out_pts_b.clone().detach().requires_grad_(True)

                l_cont, l_mom = self.physics.compute_pde_loss(self.model, c_int)
                pde_loss = (100.0 * l_cont) + l_mom
                
                preds_int = self.model(c_int)
                l_pos = torch.mean(torch.relu(-preds_int[:, 0:1]))
                l_radial = self._compute_radial_guide_loss(preds_int, c_int)
                
                l_wall = torch.mean(self.model(c_bc)[:, 0:3]**2)
                l_in = self._compute_inlet_loss(self.model(c_in), c_in)
                l_out = torch.mean(self.model(c_out)[:, 3:4]**2)
                
                total = pde_loss + (1000.0 * l_radial) + (500.0 * l_pos) + self.lambda_bc * (l_wall + l_in + l_out)
                total.backward()
                return total

        lbfgs_optimizer.step(closure)
        torch.save(self.model.state_dict(), "checkpoints/pipe_flow_final.pth")
        logger.info("Training completed successfully.")
        return history
