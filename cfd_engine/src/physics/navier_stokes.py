import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)


class NavierStokes3DPhysics:
    """
    3D sıkıştırılamaz Navier-Stokes fizik motoru.

    Makale bağlantısı (Vurdem, Multi-Scale Fourier Feature Embeddings):
    - Eq. (2):  K(v_i, v_j) = <∇_θ u_θ(v_i), ∇_θ u_θ(v_j)>  → compute_empirical_ntk()
    - Bölüm 3.3: K = J J^T empirik NTK diagnostik       → compute_ntk_spectral_metrics()
    - PDE residual: ∇_v u_θ (uzaysal türev)              → compute_pde_residuals() / compute_pde_loss()

    Not: Eq. (2) parametre uzayındaki gradyan etkileşimini tanımlar; compute_pde_loss ise
    fizik kısıtı için koordinat uzayındaki otomatik türev zincirini kurar. İkisi farklı
    kavramlardır, ancak eğitim sırasında loss → ∇_θ R_θ zinciri kopmamalıdır.
    """

    def __init__(
        self,
        Re: float,
        spatial_weight_start: float = 1.0,
        spatial_weight_slope: float = 1.0,
        pipe_length: float = 3.0,
        outlet_suction_strength: float = 5.0,
        outlet_suction_width: float = 0.05,
        artificial_pressure_gradient: float = 0.0,
    ):
        self.Re = Re
        self.nu = 1.0 / Re
        if self.Re > 2000:
            logger.warning(
                "Steady-State solver is not physically valid for turbulent flows (Re > 2000). "
                "Expect non-convergence or add time (t) dependency."
            )
        # Spatial weighting for curriculum learning (weights applied to PDE residuals)
        self.spatial_weight_start = spatial_weight_start
        self.spatial_weight_slope = spatial_weight_slope
        self.pipe_length = float(pipe_length)
        self.outlet_suction_strength = outlet_suction_strength
        self.outlet_suction_width = float(outlet_suction_width)
        # Small artificial pressure gradient (dP/dx) to bias flow towards +x during curriculum
        self.artificial_pressure_gradient = artificial_pressure_gradient

    def _ensure_differentiable_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if not coords.requires_grad:
            coords = coords.detach().clone().requires_grad_(True)
        return coords

    def _spatial_grad(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """∇_v u: koordinat uzayında kısmi türev (PDE residual zinciri)."""
        return torch.autograd.grad(
            outputs,
            inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]

    def compute_pde_residuals(
        self, model: nn.Module, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Navier-Stokes PDE rezidüellerini döndürür (MSE öncesi ham değerler).
        Autograd zinciri: θ → u_θ(v) → ∇_v u_θ → R_θ(v)  (create_graph=True ile kopmaz).
        """
        coords = self._ensure_differentiable_coords(coords)
        preds = model(coords)
        u, v, w, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

        grad_u = self._spatial_grad(u, coords)
        u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

        grad_v = self._spatial_grad(v, coords)
        v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]

        grad_w = self._spatial_grad(w, coords)
        w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

        grad_p = self._spatial_grad(p, coords)
        p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

        u_xx = self._spatial_grad(u_x, coords)[:, 0:1]
        u_yy = self._spatial_grad(u_y, coords)[:, 1:2]
        u_zz = self._spatial_grad(u_z, coords)[:, 2:3]
        laplacian_u = u_xx + u_yy + u_zz

        v_xx = self._spatial_grad(v_x, coords)[:, 0:1]
        v_yy = self._spatial_grad(v_y, coords)[:, 1:2]
        v_zz = self._spatial_grad(v_z, coords)[:, 2:3]
        laplacian_v = v_xx + v_yy + v_zz

        w_xx = self._spatial_grad(w_x, coords)[:, 0:1]
        w_yy = self._spatial_grad(w_y, coords)[:, 1:2]
        w_zz = self._spatial_grad(w_z, coords)[:, 2:3]
        laplacian_w = w_xx + w_yy + w_zz

        continuity = u_x + v_y + w_z
        # Apply a small artificial pressure gradient (acts like a body force) to push flow downstream
        # Note: momentum equation includes +p_x term; subtracting artificial_pressure_gradient biases rightward flow
        # Strong artificial pressure gradient used as a downstream biasing force.
        # Multiplying the pressure gradient by a factor helps force the solution
        # to propagate flow all the way to the pipe exit.
        momentum_x = (u * u_x + v * u_y + w * u_z) + p_x - self.nu * laplacian_u - 5.0 * self.artificial_pressure_gradient
        momentum_y = (u * v_x + v * v_y + w * v_z) + p_y - self.nu * laplacian_v
        momentum_z = (u * w_x + v * w_y + w * w_z) + p_z - self.nu * laplacian_w

        return continuity, momentum_x, momentum_y, momentum_z, preds

    def compute_pde_loss(self, model: nn.Module, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """PDE rezidüellerinin MSE kaybı.

        Applies a spatial weighting mask that increases the PDE penalty for points downstream
        (x > spatial_weight_start) according to spatial_weight_slope. This implements a
        simple curriculum that focuses the optimizer on later sections of the pipe.
        """
        continuity, momentum_x, momentum_y, momentum_z, _ = self.compute_pde_residuals(model, coords)

        # Spatial weighting based on X coordinate
        x = coords[:, 0:1]
        # base downstream curriculum weight
        base_weight = 1.0 + self.spatial_weight_slope * torch.clamp((x - float(self.spatial_weight_start)), min=0.0)

        # enforce PDE more strongly in the front and back pipe segments
        edge_mask = 1.0 + 4.0 * (
            torch.sigmoid((1.0 - x) * 20.0) + torch.sigmoid((x - 2.0) * 20.0)
        )
        weight = base_weight * edge_mask

        # additional continuity emphasis at the outlet location
        outlet_bias = 1.0 + self.outlet_suction_strength * torch.sigmoid(
            (x - (self.pipe_length - self.outlet_suction_width)) * 50.0
        )

        loss_cont = torch.mean(weight * outlet_bias * (continuity ** 2))
        loss_mom = torch.mean(weight * (momentum_x ** 2 + momentum_y ** 2 + momentum_z ** 2))
        return loss_cont, loss_mom

    def _param_jacobian_row(
        self, model: nn.Module, coord: torch.Tensor, scalar_fn
    ) -> torch.Tensor:
        """Tek bir collocation noktasında ∇_θ skaler çıktının flatten edilmiş Jacobian satırı."""
        params = [p for p in model.parameters() if p.requires_grad]
        model.zero_grad(set_to_none=True)
        scalar = scalar_fn(model, coord)
        grads = torch.autograd.grad(
            scalar,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        return torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device) for g, p in zip(grads, params)]
        )

    def compute_empirical_ntk(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        output_index: int = 0,
        max_points: int = 64,
    ) -> torch.Tensor:
        """
        Makale Eq. (2) ve Bölüm 3.3:
            K(v_i, v_j) = <∇_θ u_θ(v_i), ∇_θ u_θ(v_j)>
            K = J J^T   (J: ağ çıktısının θ'ya göre Jacobian'ı)

        Args:
            output_index: 0=u, 1=v, 2=w, 3=p bileşeni için NTK hesaplanır.
            max_points:   Hesaplama maliyetini sınırlamak için alt örnekleme.
        """
        if coords.shape[0] > max_points:
            idx = torch.randperm(coords.shape[0], device=coords.device)[:max_points]
            coords = coords[idx].detach()

        was_training = model.training
        model.eval()

        def scalar_output(m: nn.Module, point: torch.Tensor) -> torch.Tensor:
            return m(point.unsqueeze(0))[0, output_index]

        rows: List[torch.Tensor] = []
        for i in range(coords.shape[0]):
            rows.append(self._param_jacobian_row(model, coords[i].detach(), scalar_output))

        J = torch.stack(rows)
        K = J @ J.T

        if was_training:
            model.train()
        return K

    def compute_ntk_spectral_metrics(self, ntk_matrix: torch.Tensor) -> Dict[str, float]:
        """
        Makale Bölüm 3.3 diagnostik metrikleri:
        - condition_number (κ): λ_max / λ_min
        - spectral_decay_slope (α): log-log uzayında eigenvalue decay eğimi
        """
        eigenvalues = torch.linalg.eigvalsh(ntk_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-12].flip(0)

        if eigenvalues.numel() < 2:
            return {"condition_number": float("nan"), "spectral_decay_slope": float("nan")}

        kappa = (eigenvalues[0] / eigenvalues[-1]).item()

        indices = np.arange(1, eigenvalues.numel() + 1, dtype=np.float64)
        log_i = np.log(indices)
        log_lambda = np.log(eigenvalues.detach().cpu().numpy())
        alpha = -float(np.polyfit(log_i, log_lambda, 1)[0])

        return {"condition_number": kappa, "spectral_decay_slope": alpha}

    def compute_bc_loss(
        self, model: nn.Module, coords: torch.Tensor, u_inlet: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        preds = model(coords)
        u, v, w = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
        eps = 1e-3

        is_inlet = (x < eps).squeeze()
        is_wall = ((torch.abs(y) > 1.0 - eps) | (torch.abs(z) > 1.0 - eps)).squeeze()

        loss_inlet = torch.tensor(0.0, device=coords.device)
        loss_wall = torch.tensor(0.0, device=coords.device)

        if is_inlet.any():
            u_in, v_in, w_in = u[is_inlet], v[is_inlet], w[is_inlet]
            loss_inlet = torch.mean((u_in - u_inlet) ** 2 + v_in ** 2 + w_in ** 2)

        if is_wall.any():
            u_w, v_w, w_w = u[is_wall], v[is_wall], w[is_wall]
            loss_wall = torch.mean(u_w ** 2 + v_w ** 2 + w_w ** 2)

        return loss_inlet, loss_wall

    def compute_pressure_pinning(self, model: nn.Module, outlet_coords: torch.Tensor, p_ref: float = 0.0) -> torch.Tensor:
        """
        Enforce pressure pinning at outlet (e.g., x = length) to remove pressure null-space.
        Returns MSE penalty on (p - p_ref).
        """
        preds = model(outlet_coords)
        p = preds[:, 3:4]
        loss_pin = torch.mean((p - p_ref) ** 2)
        return loss_pin

    def compute_positivity_loss(self, model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        """
        Penalize negative streamwise velocity `u` inside domain to avoid flow collapse/backflow.
        Uses a soft penalty: mean(ReLU(-u))^2 so optimizer can correct negatives smoothly.
        """
        preds = model(coords)
        u = preds[:, 0:1]
        neg = torch.relu(-u)
        loss_pos = torch.mean(neg ** 2)
        return loss_pos
