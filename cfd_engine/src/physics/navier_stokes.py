"""
3D Incompressible Navier-Stokes Physics Engine.

NS equations (no artificial body forces):
  Continuity:  du/dx + dv/dy + dw/dz = 0
  Momentum X:  u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx+u_yy+u_zz) = 0
  Momentum Y:  u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx+v_yy+v_zz) = 0
  Momentum Z:  u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx+w_yy+w_zz) = 0

Speed optimisation:
  First-order  derivatives → create_graph=True  (needed for total_loss.backward)
  Second-order derivatives → create_graph=False (avoids 3rd-order graph, ~3-4x faster)

compute_pde_residuals returns 6 values:
  (continuity, mom_x, mom_y, mom_z, preds, u_x)
  u_x is returned so the trainer can compute axial smoothness loss
  mean(u_x²) without an extra forward pass.
"""
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class NavierStokes3DPhysics:

    def __init__(self, Re: float, pipe_length: float = 3.0):
        self.Re = Re
        self.nu = 1.0 / Re
        self.pipe_length = float(pipe_length)
        if Re > 2000:
            logger.warning(
                "Steady solver not valid for turbulent flow (Re > 2000)."
            )

    def _grad(
        self,
        outputs: torch.Tensor,
        inputs: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Spatial partial derivative via autograd.

        retain_graph=True always — the full computation graph must stay alive
        until total_loss.backward() has finished traversing all branches.
        """
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=create_graph,
            retain_graph=True,
        )[0]

    def compute_pde_residuals(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        second_order_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (continuity, mom_x, mom_y, mom_z, preds, u_x)

        second_order_graph:
          False (Adam stage) — 2nd derivatives use create_graph=False.
            Fast (~3-4×) but the Laplacian's gradient w.r.t. θ is NOT
            backpropagated, so momentum residual cannot be fully minimised
            through the viscous term.
          True  (L-BFGS stage) — 2nd derivatives use create_graph=True.
            Slower, but the full momentum equation (including ν∇²u) is
            differentiable, letting L-BFGS drive momentum residual to ~0.

        u_x (= du/dx) is returned as 6th element so the trainer can compute
        the axial smoothness loss mean(u_x²) for free.
        """
        if not coords.requires_grad:
            coords = coords.detach().clone().requires_grad_(True)

        preds = model(coords)
        u, v, w, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

        # ── first-order (create_graph=True — must stay on graph) ─────────────
        gu = self._grad(u, coords)
        u_x, u_y, u_z = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]

        gv = self._grad(v, coords)
        v_x, v_y, v_z = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]

        gw = self._grad(w, coords)
        w_x, w_y, w_z = gw[:, 0:1], gw[:, 1:2], gw[:, 2:3]

        gp = self._grad(p, coords)
        p_x, p_y, p_z = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]

        # ── second-order ─────────────────────────────────────────────────────
        # create_graph controlled by second_order_graph:
        #   Adam → False (fast)   |   L-BFGS → True (full momentum optimisation)
        cg2 = second_order_graph
        u_xx = self._grad(u_x, coords, create_graph=cg2)[:, 0:1]
        u_yy = self._grad(u_y, coords, create_graph=cg2)[:, 1:2]
        u_zz = self._grad(u_z, coords, create_graph=cg2)[:, 2:3]

        v_xx = self._grad(v_x, coords, create_graph=cg2)[:, 0:1]
        v_yy = self._grad(v_y, coords, create_graph=cg2)[:, 1:2]
        v_zz = self._grad(v_z, coords, create_graph=cg2)[:, 2:3]

        w_xx = self._grad(w_x, coords, create_graph=cg2)[:, 0:1]
        w_yy = self._grad(w_y, coords, create_graph=cg2)[:, 1:2]
        w_zz = self._grad(w_z, coords, create_graph=cg2)[:, 2:3]

        laplacian_u = u_xx + u_yy + u_zz
        laplacian_v = v_xx + v_yy + v_zz
        laplacian_w = w_xx + w_yy + w_zz

        # ── NS residuals (no body force) ──────────────────────────────────────
        continuity = u_x + v_y + w_z
        momentum_x = u * u_x + v * u_y + w * u_z + p_x - self.nu * laplacian_u
        momentum_y = u * v_x + v * v_y + w * v_z + p_y - self.nu * laplacian_v
        momentum_z = u * w_x + v * w_y + w * w_z + p_z - self.nu * laplacian_w

        # u_x returned for axial smoothness (free — already computed above)
        return continuity, momentum_x, momentum_y, momentum_z, preds, u_x

    # ── Diagnostic helpers (not called during training) ───────────────────────

    def _param_jacobian_row(self, model, coord, scalar_fn):
        params = [p for p in model.parameters() if p.requires_grad]
        model.zero_grad(set_to_none=True)
        scalar = scalar_fn(model, coord)
        grads = torch.autograd.grad(
            scalar, params,
            retain_graph=False, create_graph=False, allow_unused=True,
        )
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, params)
        ])

    def compute_empirical_ntk(
        self, model, coords, output_index=0, max_points=64,
    ) -> torch.Tensor:
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
            coords = coords[idx].detach()
        was_training = model.training
        model.eval()
        def scalar_output(m, pt):
            return m(pt.unsqueeze(0))[0, output_index]
        rows = [
            self._param_jacobian_row(model, coords[i].detach(), scalar_output)
            for i in range(coords.shape[0])
        ]
        K = torch.stack(rows) @ torch.stack(rows).T
        if was_training:
            model.train()
        return K

    def compute_ntk_spectral_metrics(self, ntk_matrix: torch.Tensor) -> Dict[str, float]:
        eigenvalues = torch.linalg.eigvalsh(ntk_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-12].flip(0)
        if eigenvalues.numel() < 2:
            return {"condition_number": float("nan"), "spectral_decay_slope": float("nan")}
        kappa = (eigenvalues[0] / eigenvalues[-1]).item()
        indices = np.arange(1, eigenvalues.numel() + 1, dtype=np.float64)
        alpha = -float(
            np.polyfit(np.log(indices),
                       np.log(eigenvalues.detach().cpu().numpy()), 1)[0]
        )
        return {"condition_number": kappa, "spectral_decay_slope": alpha}
