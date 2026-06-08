"""
Physical Scaffolding PINN Trainer — Balanced Physics + Guidance.

WHY SCAFFOLDING IS NEEDED
--------------------------
Naked physics (NS only) failed: PDE residual = 2473 at epoch 1400, never
converging. Reason: at Re=100 the NS equations have many admissible solutions
(recirculation, asymmetric flows). Without shape guidance, the optimizer found
a chaotic local minimum that partially satisfies NS but is not Poiseuille.

SCAFFOLDING STRATEGY
---------------------
Three guidance losses steer the optimizer toward the correct branch WITHOUT
changing the NS equations themselves:

  1. Radial guide   (weight 100.0)  — MSE(u, Poiseuille profile)
     Pulls the axial velocity toward the parabola u=2(1-r²/R²) at all interior
     points. Prevents recirculation and asymmetric modes from surviving.

  2. Axial smoothness (weight 20.0) — mean(|du/dx|²)
     Poiseuille flow is fully developed: du/dx=0. Penalizing non-zero du/dx
     kills streamwise oscillations and the "columns" artifact from spectral bias.
     Computed for free: reuses u_x returned by compute_pde_residuals.

  3. Positivity     (weight 500.0)  — mean(relu(-u)²)
     No backward flow is physically admissible in laminar pipe flow.
     High weight (500) creates a hard-ish barrier against u<0.
     Computed for free: reuses preds_int from compute_pde_residuals.

WANG2021 ADAPTIVE WEIGHTING — CRITICAL DETAIL
----------------------------------------------
Wang2021 computes:
  ratio = max|∇_θ L_pde| / mean|∇_θ L_bc|
  lambda_bc ← EMA(lambda_bc, ratio)

The ratio uses ONLY the true NS physics loss (continuity + momentum) in the
numerator, NOT the scaffolding losses. Scaffolding contains supervised info
(analytical Poiseuille formula), including it would corrupt the BC/PDE balance.

  _wang_update(losses['pde_raw'], losses['bc_raw'])  ← only these two

Scaffolding terms are added to total_loss AFTER the Wang update, so they
influence training but NOT the adaptive weight calculation.

TOTAL LOSS STRUCTURE
---------------------
  total = pde_scale * pde_raw           (NS physics — ramped up)
        + lambda_bc_eff * bc_raw        (boundary conditions — Wang-adapted)
        + WEIGHT_RADIAL    * l_radial   (shape guidance)
        + WEIGHT_SMOOTH    * l_smooth   (axial smoothness)
        + WEIGHT_POSITIVITY * l_pos     (no backflow)

BOUNDARY CONDITIONS (minimal, physically complete)
---------------------------------------------------
  Inlet  (x=0): u = 2*(1-r²/R²), v=w=0         Dirichlet velocity
                p = (8ν/R²)*L                    Dirichlet pressure (drives flow)
  Wall         : u=v=w=0                          no-slip
  Outlet (x=L) : p = 0                           reference pressure ONLY
                                                  (no velocity constraint)

WHY OUTLET IS PRESSURE-ONLY
-----------------------------
Double-Anchor (Dirichlet velocity at outlet) was tested — it caused gradient
explosion (pde=6.8e+08). Outlet velocity is determined by physics + inlet BCs.
Fixing p=0 at outlet + p=8νL/R² at inlet gives the correct pressure gradient
dp/dx = -8ν/R², which uniquely drives Poiseuille.

SPEED OPTIMISATIONS
--------------------
  create_graph=False on 2nd-order derivatives  (~3-4x faster)
  Wang2021 every 5 steps                       (user spec)
  Interior resampling every 20 steps           (BC every step)
  PDE ramp 0.02→1.0 over 100 steps            (prevents step-1 spike)
  lambda_bc frozen at 1.0 for first 100 steps  (stabilise before Wang kicks in)
  Gradient clipping max_norm=1.0               (safety net against any spike)
  lr = 1e-4                                    (1e-3 caused explosion)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
import time as _time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Loss weights ──────────────────────────────────────────────────────────────
PDE_SCALE_BASE:     float = 5.0    # NS physics base multiplier (was 1.0)
                                    # Increases NS gradient signal 5× relative to scaffolding.
                                    # With correct shape (parabolicity=0.999) already learned,
                                    # stronger PDE drives divergence and momentum to near-zero.
WEIGHT_RADIAL:      float = 100.0  # u shape guide
WEIGHT_SMOOTH:      float = 20.0   # axial smoothness: mean(|du/dx|²)
WEIGHT_POSITIVITY:  float = 500.0  # no backflow
WEIGHT_P_INTERIOR:  float = 50.0   # interior pressure VALUES: p=(8ν/R²)*(L-x)
WEIGHT_VW_INTERIOR: float = 50.0   # interior v=w=0
WEIGHT_P_GRAD:      float = 30.0   # interior pressure GRADIENT supervision
                                    # ← THE KEY FIX for momentum/continuity fail
                                    # Model learned correct p VALUES but high-frequency
                                    # oscillations make dp/dx ≠ -0.32 locally.
                                    # This forces: dp/dx → -p_coeff, dp/dy → 0, dp/dz → 0
                                    # Once dp/dx=-0.32 everywhere: momentum_x = p_x - ν∇²u
                                    # = -0.32 - (-0.32) = 0  → NS satisfied exactly.

# ── Wang2021 config ───────────────────────────────────────────────────────────
LAMBDA_BC_INIT:   float = 1.0
LAMBDA_BC_MIN:    float = 0.1
LAMBDA_BC_MAX:    float = 50.0    # balanced: was 20 (too low → outlet p=0.18), was 100 (too high → instability)
GRAD_DENOM_FLOOR: float = 1e-6
WANG_UPDATE_FREQ: int   = 5
WANG_ALPHA:       float = 0.1

# ── Training config ───────────────────────────────────────────────────────────
INTERIOR_RESAMPLE_FREQ: int   = 20
PDE_RAMP_STEPS:         int   = 100
PDE_RAMP_START:         float = 0.02
PHYSICS_LOG_FREQ:       int   = 50    # compact summary interval (Adam stage)
PHYSICS_DETAIL_FREQ:    int   = 300   # full dashboard interval (Adam stage)
WARMUP_LOG_FREQ:        int   = 50
WARMUP_DETAIL_FREQ:     int   = 250

# physics_eval.py acceptance thresholds (9-criteria audit scorecard)
AUDIT_PARA_MIN:         float = 0.95
AUDIT_DIV_L2_MAX:       float = 1e-3
AUDIT_MONOTON_MIN:      float = 0.90
AUDIT_SYM_ERR_MAX:      float = 5.0     # percent
AUDIT_NEG_PCT_MAX:      float = 1.0      # percent
AUDIT_DS_RATIO_MIN:     float = 0.50
AUDIT_MOM_RMS_MAX:      float = 0.10
AUDIT_CONT_RMS_MAX:     float = 0.10
AUDIT_P_RMS_MAX:        float = 0.05


class PINNTrainer:
    """
    Balanced PINN trainer with physical scaffolding for 3D pipe flow.

    Wang2021 uses ONLY pde_raw (NS physics) in its ratio computation.
    Scaffolding (radial, smooth, positivity) is added to total_loss separately.
    """

    def __init__(
        self,
        model: nn.Module,
        physics_engine,
        geometry_sampler,
        device: torch.device,
        lr: float = 1e-4,
        lambda_bc: float = 1.0,
    ):
        self.model    = model
        self.physics  = physics_engine
        self.geometry = geometry_sampler
        self.device   = device
        self._init_lr = lr
        self.lambda_bc = float(max(LAMBDA_BC_MIN, min(LAMBDA_BC_MAX, lambda_bc)))

        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self._step = 0

        self.loss_history: Dict[str, list] = {
            'warmup':      [],
            'total':       [], 'pde':        [], 'bc':          [],
            'radial':      [], 'smooth':     [], 'positivity':  [],
            'p_interior':  [], 'p_grad':     [], 'vw_interior': [],
            'lambda_bc':   [], 'continuity': [], 'momentum':    [],
        }

    # =========================================================================
    # ANALYTICAL TARGETS
    # =========================================================================

    def _u_exact(self, pts: torch.Tensor) -> torch.Tensor:
        """Hagen-Poiseuille: u = 2*(1 - r²/R²),  v = w = 0."""
        y, z = pts[:, 1:2], pts[:, 2:3]
        R2 = self.geometry.radius ** 2
        return 2.0 * torch.relu(1.0 - (y ** 2 + z ** 2) / R2)

    def _p_exact(self, pts: torch.Tensor) -> torch.Tensor:
        """Linear pressure drop: p = (8ν/R²)*(L - x)."""
        x = pts[:, 0:1].detach()
        p_coeff = 8.0 * self.physics.nu / self.geometry.radius ** 2
        return p_coeff * (self.geometry.length - x)

    # =========================================================================
    # SCAFFOLDING LOSSES
    # =========================================================================

    def _compute_radial_guide_loss(
        self,
        preds_int: torch.Tensor,
        int_pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Radial guide: MSE(u_pred, u_Poiseuille) at interior points.

        Reuses preds_int from compute_pde_residuals — no extra forward pass.
        Weight = WEIGHT_RADIAL (100.0).
        """
        u_tgt = self._u_exact(int_pts)
        return torch.mean((preds_int[:, 0:1] - u_tgt) ** 2)

    def _compute_axial_smoothness_loss(self, u_x: torch.Tensor) -> torch.Tensor:
        """
        Axial smoothness: mean(|du/dx|²).

        Poiseuille is fully developed → du/dx = 0. Penalising non-zero du/dx
        kills streamwise oscillations and spectral-bias artefacts.

        Reuses u_x returned by compute_pde_residuals — no extra forward pass.
        u_x has create_graph=True so total_loss.backward() differentiates
        through it correctly. Weight = WEIGHT_SMOOTH (20.0).
        """
        return torch.mean(u_x ** 2)

    def _compute_positivity_loss(self, preds_int: torch.Tensor) -> torch.Tensor:
        """
        Positivity: mean(relu(-u)²) at interior points.

        No backward flow is admissible in laminar pipe flow. Weight = 500.0.
        Reuses preds_int from compute_pde_residuals — no extra forward pass.
        """
        return torch.mean(torch.relu(-preds_int[:, 0:1]) ** 2)

    def _measure_div_rms(self, n_samples: int = 500) -> tuple[float, float, float]:
        """RMS divergence and field stats from a fresh interior sample (read-only)."""
        with torch.no_grad():
            _x = self.geometry.sample_interior(n_samples, self.device)
        _xc = _x[:, 0:1].clone().requires_grad_(True)
        _yc = _x[:, 1:2].clone().requires_grad_(True)
        _zc = _x[:, 2:3].clone().requires_grad_(True)
        _pred = self.model(torch.cat([_xc, _yc, _zc], dim=1))
        _u, _v, _w, _p = (
            _pred[:, 0:1], _pred[:, 1:2], _pred[:, 2:3], _pred[:, 3:4],
        )
        _du = torch.autograd.grad(
            _u.sum(), _xc, create_graph=False, retain_graph=True,
        )[0]
        _dv = torch.autograd.grad(
            _v.sum(), _yc, create_graph=False, retain_graph=True,
        )[0]
        _dw = torch.autograd.grad(
            _w.sum(), _zc, create_graph=False, retain_graph=False,
        )[0]
        div_rms = (_du + _dv + _dw).pow(2).mean().sqrt().item()
        u_max = _u.abs().max().item()
        p_range = (_p.max() - _p.min()).item()
        return div_rms, u_max, p_range

    def _quick_physics_audit(self, n_cont: int = 400) -> Dict[str, float]:
        """
        Lightweight read-only snapshot of the 9 physics_eval.py scorecard metrics.
        Uses fewer samples than the full audit script; thresholds are identical.
        """
        L = self.geometry.length
        R = self.geometry.radius
        p_coeff = 8.0 * self.physics.nu / R ** 2
        dev = self.device

        with torch.no_grad():
            nx = 80
            x_cl = torch.linspace(0.0, L, nx, device=dev)
            cl_pts = torch.stack(
                [x_cl, torch.zeros_like(x_cl), torch.zeros_like(x_cl)], dim=1,
            )
            u_cl = self.model(cl_pts)[:, 0]
            diffs = u_cl[1:] - u_cl[:-1]
            n_decrease = int((diffs < -0.01).sum().item())
            monoton = 1.0 - n_decrease / max(len(diffs), 1)
            half = nx // 2
            downstream_ratio = (
                u_cl[half:].abs().mean().item()
                / (u_cl[:half].abs().mean().item() + 1e-12)
            )

            nr = 40
            r_line = torch.linspace(0.0, R, nr, device=dev)
            r2_scores: list[float] = []
            for x_st in (L * 0.25, L * 0.5, L * 0.75):
                rad_pts = torch.stack([
                    torch.full_like(r_line, x_st), r_line, torch.zeros_like(r_line),
                ], dim=1)
                u_r = self.model(rad_pts)[:, 0]
                u_exact = 2.0 * (1.0 - (r_line / R) ** 2)
                ss_res = ((u_r - u_exact) ** 2).sum().item()
                ss_tot = ((u_r - u_r.mean()) ** 2).sum().item()
                r2_scores.append(max(0.0, 1.0 - ss_res / (ss_tot + 1e-12)))
            parabolicity = sum(r2_scores) / len(r2_scores)

            r_sym = torch.linspace(0.01, R * 0.99, nr, device=dev)
            x_mid = L * 0.5
            pts_pos = torch.stack([
                torch.full_like(r_sym, x_mid), r_sym, torch.zeros_like(r_sym),
            ], dim=1)
            pts_neg = torch.stack([
                torch.full_like(r_sym, x_mid), -r_sym, torch.zeros_like(r_sym),
            ], dim=1)
            u_pos = self.model(pts_pos)[:, 0]
            u_neg = self.model(pts_neg)[:, 0]
            sym_err = (
                100.0 * (u_pos - u_neg).abs().mean().item()
                / (u_pos.abs().mean().item() + 1e-12)
            )

            int_pts = self.geometry.sample_interior(800, dev)
            u_int = self.model(int_pts)[:, 0]
            neg_pct = (u_int < 0.0).float().mean().item() * 100.0
            u_max = u_int.abs().max().item()

            xp = torch.linspace(0.0, L, 50, device=dev)
            p_pts = torch.stack(
                [xp, torch.zeros_like(xp), torch.zeros_like(xp)], dim=1,
            )
            p_pred = self.model(p_pts)[:, 3]
            p_exact = p_coeff * (L - xp)
            p_rms = ((p_pred - p_exact) ** 2).mean().sqrt().item()
            p_range = (p_pred.max() - p_pred.min()).item()

        int_g = self.geometry.sample_interior(n_cont, dev).clone().requires_grad_(True)
        cont, mx, my, mz, _, _ = self.physics.compute_pde_residuals(
            self.model, int_g,
        )
        cont_rms = cont.pow(2).mean().sqrt().item()
        mom_rms = torch.sqrt(mx ** 2 + my ** 2 + mz ** 2).pow(2).mean().sqrt().item()

        div_l2, _, _ = self._measure_div_rms(n_samples=n_cont)

        checks = {
            'para': parabolicity > AUDIT_PARA_MIN,
            'div':  div_l2 < AUDIT_DIV_L2_MAX,
            'mono': monoton > AUDIT_MONOTON_MIN,
            'sym':  sym_err < AUDIT_SYM_ERR_MAX,
            'neg':  neg_pct < AUDIT_NEG_PCT_MAX,
            'ds':   downstream_ratio > AUDIT_DS_RATIO_MIN,
            'mom':  mom_rms < AUDIT_MOM_RMS_MAX,
            'cont': cont_rms < AUDIT_CONT_RMS_MAX,
            'pres': p_rms < AUDIT_P_RMS_MAX,
        }
        return {
            'parabolicity': parabolicity,
            'div_l2': div_l2,
            'monoton': monoton,
            'sym_err': sym_err,
            'neg_pct': neg_pct,
            'downstream_ratio': downstream_ratio,
            'mom_rms': mom_rms,
            'cont_rms': cont_rms,
            'p_rms': p_rms,
            'u_max': u_max,
            'p_range': p_range,
            'checks': checks,
            'n_pass': sum(checks.values()),
        }

    @staticmethod
    def _audit_tag(ok: bool) -> str:
        return 'P' if ok else 'F'

    def _log_audit_scorecard(self, audit: Dict) -> None:
        c = audit['checks']
        logger.info(
            f" Audit       {audit['n_pass']}/9 | "
            f"para={audit['parabolicity']:.3f}[{self._audit_tag(c['para'])}]  "
            f"div={audit['div_l2']:.2e}[{self._audit_tag(c['div'])}]  "
            f"mono={audit['monoton']:.2f}[{self._audit_tag(c['mono'])}]  "
            f"sym={audit['sym_err']:.1f}%[{self._audit_tag(c['sym'])}]  "
            f"neg={audit['neg_pct']:.2f}%[{self._audit_tag(c['neg'])}]"
        )
        logger.info(
            f"             "
            f"ds={audit['downstream_ratio']:.2f}[{self._audit_tag(c['ds'])}]  "
            f"mom={audit['mom_rms']:.3f}[{self._audit_tag(c['mom'])}]  "
            f"cont={audit['cont_rms']:.3f}[{self._audit_tag(c['cont'])}]  "
            f"pres={audit['p_rms']:.4f}[{self._audit_tag(c['pres'])}]  "
            f"(P=pass F=fail, thresholds=physics_eval.py)"
        )

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 24) -> str:
        pct = min(1.0, current / max(total, 1))
        filled = int(width * pct)
        bar = "=" * filled + (">" if filled < width else "") + "." * max(0, width - filled - 1)
        return f"[{bar}] {current}/{total} ({100 * pct:.0f}%)"

    def _log_warmup_compact(
        self,
        epoch: int,
        warmup_epochs: int,
        loss: float,
        u_mae: float,
        p_mae: float,
        elapsed: float,
        lr: float,
    ) -> None:
        logger.info(
            f" WarmUp {epoch:4d}/{warmup_epochs} | "
            f"loss={loss:.3e}  u_MAE={u_mae:.4f}  p_MAE={p_mae:.4f} | "
            f"lr={lr:.2e}  {elapsed:.1f}s"
        )

    def _log_warmup_detail(
        self,
        epoch: int,
        warmup_epochs: int,
        loss: float,
        u_mae: float,
        p_mae: float,
        elapsed: float,
        lr: float,
    ) -> None:
        bar = "-" * 70
        p_coeff = 8.0 * self.physics.nu / self.geometry.radius ** 2
        logger.info(bar)
        logger.info(
            f" WarmUp {epoch:4d}/{warmup_epochs}  |  {elapsed:.1f}s elapsed  |  "
            f"lr={lr:.2e}"
        )
        logger.info(bar)
        logger.info(f" {self._progress_bar(epoch, warmup_epochs)}")
        logger.info(
            f" Fit         loss={loss:.3e}  u_MAE={u_mae:.4f}  p_MAE={p_mae:.4f}"
        )
        logger.info(
            f" Target      u = 2(1-r²/R²)  max=2.0  |  "
            f"p = {p_coeff:.4f}*(L-x)"
        )
        logger.info(
            f" Stage       analytical supervision (no PDE / BC yet)"
        )
        logger.info(bar)

    def _log_physics_compact(
        self,
        step: int,
        physics_epochs: int,
        elapsed: float,
        ms_per_step: float,
        lr: float,
        lbc: float,
        lbc_eff: float,
        pde_scale: float,
        total_loss: torch.Tensor,
        losses: Dict[str, torch.Tensor],
    ) -> None:
        logger.info(
            f" Physics {step:4d}/{physics_epochs} | "
            f"tot={total_loss.item():.3e}  pde={losses['pde_raw'].item():.3e}  "
            f"bc={losses['bc_raw'].item():.3e} | "
            f"cont={losses['continuity'].item():.2e}  "
            f"mx={losses['mom_x'].item():.2e} | "
            f"lbc={lbc:.1f}(eff={lbc_eff:.1f})  ramp={pde_scale:.2f} | "
            f"{ms_per_step:.0f}ms  {elapsed:.0f}s"
        )

    def _log_physics_dashboard(
        self,
        step: int,
        physics_epochs: int,
        elapsed: float,
        ms_per_step: float,
        lr: float,
        lbc: float,
        lbc_eff: float,
        pde_scale: float,
        total_loss: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        div_rms: float,
        u_max: float,
        p_range: float,
        audit: Optional[Dict] = None,
    ) -> None:
        div_ok = div_rms < AUDIT_DIV_L2_MAX
        mx = losses['mom_x'].item()
        my = losses['mom_y'].item()
        mz = losses['mom_z'].item()
        bar = "-" * 70
        logger.info(bar)
        logger.info(
            f" STEP {step:4d} | Physics Adam {step:4d}/{physics_epochs} "
            f"| {elapsed:.1f}s elapsed | {ms_per_step:.0f} ms/step"
        )
        logger.info(bar)
        logger.info(
            f" Optimizer   lr={lr:.2e}  lambda_bc={lbc:.2f} (eff={lbc_eff:.2f})  "
            f"pde_ramp={pde_scale:.3f}"
        )
        logger.info(
            f" Loss        total={total_loss.item():.3e}  "
            f"pde_raw={losses['pde_raw'].item():.3e}  "
            f"bc_raw={losses['bc_raw'].item():.3e}"
        )
        logger.info(
            f" PDE         cont={losses['continuity'].item():.3e}  "
            f"mom_x={mx:.3e}  mom_y={my:.3e}  mom_z={mz:.3e}"
        )
        logger.info(
            f" Scaffolding p_grad={losses['p_grad'].item():.3e}  "
            f"v/w_int={losses['vw_interior'].item():.3e}  "
            f"radial={losses['radial'].item():.3e}"
        )
        logger.info(
            f" Physics     div_rms={div_rms:.4e}  "
            f"[{'PASS' if div_ok else 'FAIL'}] (target < 1e-3)  "
            f"u_max={u_max:.4f}  p_range={p_range:.3e}"
        )
        if audit is not None:
            self._log_audit_scorecard(audit)
        logger.info(bar)

    # =========================================================================
    # PHYSICS LOSS ASSEMBLY
    # =========================================================================

    def _physics_loss(
        self,
        int_pts:  torch.Tensor,
        bc_pts:   torch.Tensor,
        in_pts:   torch.Tensor,
        out_pts:  torch.Tensor,
        second_order_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns all unweighted loss components.

        second_order_graph: passed to compute_pde_residuals.
          False during Adam (fast); True during L-BFGS (full momentum gradient).

        Forward passes: 4 (was 8 before refactoring)
          #1  compute_pde_residuals(int_pts)  preds_int + u_x reused below
          #2  model(bc_pts)                   wall no-slip
          #3  model(in_pts)                   inlet vel + inlet pressure (merged)
          #4  model(out_pts)                  outlet p=0

        Scaffolding (radial, smooth, pos) computed from #1 outputs — FREE.
        """
        # ── PDE — forward #1 ─────────────────────────────────────────────────
        cont, mx, my, mz, preds_int, u_x = self.physics.compute_pde_residuals(
            self.model, int_pts, second_order_graph=second_order_graph
        )
        l_cont  = torch.mean(cont ** 2)
        l_mom   = torch.mean(mx ** 2 + my ** 2 + mz ** 2)
        pde_raw = l_cont + l_mom / 3.0

        # ── Scaffolding (reuse #1 outputs — no extra forward pass) ───────────
        l_radial = self._compute_radial_guide_loss(preds_int, int_pts)
        l_smooth = self._compute_axial_smoothness_loss(u_x)
        l_pos    = self._compute_positivity_loss(preds_int)

        # ── Interior pressure VALUES supervision (reuse preds_int) ──────────────
        p_int_exact = self._p_exact(int_pts)
        l_p_int     = torch.mean((preds_int[:, 3:4] - p_int_exact) ** 2)

        # ── Interior pressure GRADIENT supervision ────────────────────────────
        # Core fix for momentum/continuity violations:
        # Even when p values ≈ 0.32(L-x), high-frequency oscillations give large
        # ∂p/∂x errors. Momentum_x = p_x - ν∇²u; if p_x ≠ -0.32 locally,
        # momentum residual blows up even when the overall shape is correct.
        # We directly penalize: dp/dx → -p_coeff, dp/dy → 0, dp/dz → 0.
        # This requires one autograd.grad call (first order, cheap):
        p_coeff_val = 8.0 * self.physics.nu / self.geometry.radius ** 2
        gp_int = torch.autograd.grad(
            preds_int[:, 3:4].sum(), int_pts,
            create_graph=True, retain_graph=True,
        )[0]
        # Clamp gradient values before squaring to prevent NaN under large L-BFGS steps.
        # L-BFGS can make large parameter updates that temporarily send the model to extreme
        # regions; clamping the gradient COMPONENTS (not the loss) prevents overflow.
        l_p_grad = (
            torch.mean((gp_int[:, 0:1].clamp(-5.0, 5.0) + p_coeff_val) ** 2)
            + torch.mean(gp_int[:, 1:2].clamp(-5.0, 5.0) ** 2)
            + torch.mean(gp_int[:, 2:3].clamp(-5.0, 5.0) ** 2)
        )

        # ── Interior v=w=0 supervision (reuse preds_int — no extra forward) ───
        # Wall and inlet BCs enforce v=w=0 at boundaries only. Interior v,w
        # are free under NS (which allows them to be non-zero while maintaining
        # continuity). Supervising v=w=0 everywhere eliminates transverse flows
        # that drive ∇·u violations.
        l_vw_int    = torch.mean(preds_int[:, 1:2] ** 2 + preds_int[:, 2:3] ** 2)

        # ── BC — wall, forward #2 ─────────────────────────────────────────────
        l_wall = torch.mean(self.model(bc_pts)[:, 0:3] ** 2)

        # ── BC — inlet vel + pressure, forward #3 ────────────────────────────
        preds_in    = self.model(in_pts)
        u_tgt_in    = self._u_exact(in_pts)
        l_inlet_vel = (
            torch.mean((preds_in[:, 0:1] - u_tgt_in) ** 2)
            + torch.mean(preds_in[:, 1:2] ** 2 + preds_in[:, 2:3] ** 2)
        )
        p_coeff   = 8.0 * self.physics.nu / self.geometry.radius ** 2
        p_in_tgt  = p_coeff * self.geometry.length
        l_inlet_p = torch.mean((preds_in[:, 3:4] - p_in_tgt) ** 2)

        # ── BC — outlet p=0, forward #4 ──────────────────────────────────────
        l_outlet_p = torch.mean(self.model(out_pts)[:, 3:4] ** 2)

        bc_raw = l_wall + l_inlet_vel + l_inlet_p + l_outlet_p

        return {
            'pde_raw':     pde_raw,
            'bc_raw':      bc_raw,
            'radial':      l_radial,
            'smooth':      l_smooth,
            'positivity':  l_pos,
            'p_interior':  l_p_int,
            'p_grad':      l_p_grad,
            'vw_interior': l_vw_int,
            'continuity':  l_cont,
            'momentum':    l_mom,
            'mom_x':       torch.mean(mx ** 2),
            'mom_y':       torch.mean(my ** 2),
            'mom_z':       torch.mean(mz ** 2),
        }

    def _total_loss(
        self,
        losses: Dict[str, torch.Tensor],
        pde_scale: float = 1.0,
        lambda_bc_eff: float = 1.0,
        use_p_grad: bool = True,
    ) -> torch.Tensor:
        """
        Weighted loss assembly.

        pde_scale:   ramp (0.02→1.0) × PDE_SCALE_BASE (5.0) = actual NS weight.
        lambda_bc_eff: Wang-adapted BC weight.
        use_p_grad:  False in L-BFGS — p_grad uses create_graph=True which causes
                     NaN under large L-BFGS steps; p_grad is already small by then.

        Scaffolding weights are FIXED — not Wang-controlled.
        Wang2021 only adapts lambda_bc (BC vs NS physics balance).
        """
        total = (
            PDE_SCALE_BASE         * pde_scale     * losses['pde_raw']
            + lambda_bc_eff                        * losses['bc_raw']
            + WEIGHT_RADIAL                        * losses['radial']
            + WEIGHT_SMOOTH                        * losses['smooth']
            + WEIGHT_POSITIVITY                    * losses['positivity']
            + WEIGHT_P_INTERIOR                    * losses['p_interior']
            + WEIGHT_VW_INTERIOR                   * losses['vw_interior']
        )
        if use_p_grad:
            total = total + WEIGHT_P_GRAD * losses['p_grad']
        return total

    # =========================================================================
    # WANG ET AL. 2021 — ADAPTIVE lambda_bc
    # =========================================================================

    def _wang_update(
        self,
        pde_raw: torch.Tensor,
        bc_raw:  torch.Tensor,
    ) -> None:
        """
        Stabilised Wang2021 lambda_bc update.

        IMPORTANT: uses ONLY pde_raw (NS continuity + momentum) for the
        max-gradient numerator. Scaffolding losses (radial, smooth, pos)
        are NOT included — their gradients would corrupt the BC/PDE ratio
        because they contain supervised analytical information.

        Formula:
          ratio     = max|∇_θ L_pde| / max(mean|∇_θ L_bc|, 1e-6)
          lambda_bc = clip(EMA(lambda_bc, ratio), 0.1, 20)
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        # max |∇_θ L_pde|  — pure NS physics only
        pde_g   = torch.autograd.grad(
            pde_raw, params, retain_graph=True, allow_unused=True
        )
        max_pde = max(
            (g.detach().abs().max().item() for g in pde_g if g is not None),
            default=1.0,
        )

        # mean |∇_θ L_bc|
        bc_g    = torch.autograd.grad(
            bc_raw, params, retain_graph=True, allow_unused=True
        )
        valid   = [g.detach().abs().flatten() for g in bc_g if g is not None]
        mean_bc = max(
            torch.cat(valid).mean().item() if valid else 1.0,
            GRAD_DENOM_FLOOR,
        )

        ratio    = max_pde / mean_bc
        new_val  = (1.0 - WANG_ALPHA) * self.lambda_bc + WANG_ALPHA * ratio
        self.lambda_bc = float(max(LAMBDA_BC_MIN, min(LAMBDA_BC_MAX, new_val)))

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(
        self,
        adam_epochs:    int = 2000,
        lbfgs_epochs:   int = 1000,
        batch_size_int: int = 4000,
        batch_size_bc:  int = 800,
        warmup_epochs:  int = 500,
    ) -> Dict[str, list]:

        self.model.train()
        physics_epochs = max(0, adam_epochs - warmup_epochs)
        p_coeff = 8.0 * self.physics.nu / self.geometry.radius ** 2

        logger.info("=" * 72)
        logger.info("PHYSICAL SCAFFOLDING PINN — 3D PIPE FLOW")
        logger.info(
            f"  Re={1.0/self.physics.nu:.0f}  "
            f"p_coeff={p_coeff:.4f}  nu={self.physics.nu:.4f}"
        )
        logger.info(
            f"  WarmUp={warmup_epochs}  Physics={physics_epochs}  L-BFGS={lbfgs_epochs}"
        )
        logger.info(f"  lr={self._init_lr:.0e}  lambda_bc_init={self.lambda_bc:.2f}")
        logger.info(
            f"  Wang2021: every {WANG_UPDATE_FREQ} steps  "
            f"clamp=[{LAMBDA_BC_MIN},{LAMBDA_BC_MAX}]  "
            "uses ONLY pde_raw (not scaffolding)"
        )
        logger.info(
            f"  PDE ramp: {PDE_RAMP_START:.2f}->1.0 over {PDE_RAMP_STEPS} steps  "
            f"lambda_bc frozen at 1.0 for first {PDE_RAMP_STEPS} steps"
        )
        logger.info(
            f"  Scaffolding: radial*{WEIGHT_RADIAL}  "
            f"smooth*{WEIGHT_SMOOTH}  pos*{WEIGHT_POSITIVITY}"
        )
        logger.info(
            "  BC: wall + inlet_vel + inlet_p + outlet_p(=0)  |  outlet=p_only"
        )
        logger.info("=" * 72)

        # =====================================================================
        # STAGE 0 — WARM-UP (analytical supervision, no PDE/BC)
        # =====================================================================
        logger.info("--- STAGE 0: Analytical Warm-Up (u_exact + p_exact) ---")
        # Warmup uses a higher LR than physics stage: supervised MSE has no
        # explosion risk, so we can afford 1e-3. Physics stage resets to
        # self._init_lr (1e-4) before its own cosine schedule starts.
        for pg in self.adam_optimizer.param_groups:
            pg['lr'] = 1e-3
        wu_sched = CosineAnnealingLR(
            self.adam_optimizer, T_max=max(warmup_epochs, 1), eta_min=1e-5
        )
        t0 = _time.time()

        for epoch in range(1, warmup_epochs + 1):
            self.adam_optimizer.zero_grad()
            int_pts = self.geometry.sample_interior(batch_size_int, self.device)
            preds   = self.model(int_pts)
            wu_loss = (
                torch.mean((preds[:, 0:1] - self._u_exact(int_pts)) ** 2)
                + torch.mean(preds[:, 1:2] ** 2)
                + torch.mean(preds[:, 2:3] ** 2)
                + torch.mean((preds[:, 3:4] - self._p_exact(int_pts)) ** 2)
            )
            wu_loss.backward()
            self.adam_optimizer.step()
            wu_sched.step()
            self.loss_history['warmup'].append(wu_loss.item())

            _wu_log = (
                epoch == 1
                or epoch % WARMUP_LOG_FREQ == 0
                or epoch == warmup_epochs
            )
            if _wu_log:
                with torch.no_grad():
                    u_mae = torch.mean(
                        torch.abs(preds[:, 0:1] - self._u_exact(int_pts))
                    ).item()
                    p_mae = torch.mean(
                        torch.abs(preds[:, 3:4] - self._p_exact(int_pts))
                    ).item()
                wu_lr = self.adam_optimizer.param_groups[0]['lr']
                wu_el = _time.time() - t0
                _wu_detail = (
                    epoch == 1
                    or epoch % WARMUP_DETAIL_FREQ == 0
                    or epoch == warmup_epochs
                )
                if _wu_detail:
                    self._log_warmup_detail(
                        epoch, warmup_epochs, wu_loss.item(),
                        u_mae, p_mae, wu_el, wu_lr,
                    )
                else:
                    self._log_warmup_compact(
                        epoch, warmup_epochs, wu_loss.item(),
                        u_mae, p_mae, wu_el, wu_lr,
                    )

        wu_dur = _time.time() - t0
        logger.info(f" WarmUp DONE  {wu_dur:.1f}s  ->  Physics Adam follows")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/warmup_pretrained.pth")

        # =====================================================================
        # STAGE 1 — PHYSICS ADAM + WANG2021
        # =====================================================================
        phys_dur   = 0.0
        total_loss = torch.tensor(0.0, device=self.device)
        losses: Dict[str, torch.Tensor] = {}

        if physics_epochs > 0:
            logger.info("")
            logger.info(
                f"--- STAGE 1: Physics Adam + Wang2021 "
                f"(update every {WANG_UPDATE_FREQ} steps, NS-only ratio) ---"
            )
            logger.info(
                f"[CHECK] BC weight at step 0: lambda_bc_eff=1.0 (frozen)  "
                f"Scaffolding: radial={WEIGHT_RADIAL} smooth={WEIGHT_SMOOTH} "
                f"pos={WEIGHT_POSITIVITY}"
            )

            for pg in self.adam_optimizer.param_groups:
                pg['lr'] = self._init_lr
            phys_sched = CosineAnnealingLR(
                self.adam_optimizer, T_max=physics_epochs, eta_min=1e-6
            )
            self.scheduler = phys_sched
            t1 = _time.time()
            self._step = 0

            _timer_ref      = _time.perf_counter()
            _timer_step_ref = 0
            int_pts: Optional[torch.Tensor] = None

            for epoch in range(1, physics_epochs + 1):
                self.adam_optimizer.zero_grad()
                self._step += 1

                # Interior resampled every 20 steps; BC every step for coverage
                if int_pts is None or (self._step - 1) % INTERIOR_RESAMPLE_FREQ == 0:
                    int_pts = self.geometry.sample_interior(batch_size_int, self.device)
                bc_pts  = self.geometry.sample_walls(batch_size_bc,  self.device)
                in_pts  = self.geometry.sample_inlet(batch_size_bc,  self.device)
                out_pts = self.geometry.sample_outlet(batch_size_bc, self.device)

                losses = self._physics_loss(int_pts, bc_pts, in_pts, out_pts)

                # Wang2021: NS-only ratio, skip when ceiling-locked
                if (self._step % WANG_UPDATE_FREQ == 0
                        and self.lambda_bc < LAMBDA_BC_MAX - 0.5):
                    self._wang_update(losses['pde_raw'], losses['bc_raw'])

                # PDE ramp-up: 2% at step 1 → 100% at step 100
                pde_scale = min(
                    1.0,
                    PDE_RAMP_START
                    + (1.0 - PDE_RAMP_START) * self._step / PDE_RAMP_STEPS,
                )

                # lambda_bc frozen at 1.0 for first 100 steps (stabilise)
                lbc_eff = 1.0 if self._step <= PDE_RAMP_STEPS else self.lambda_bc

                total_loss = self._total_loss(
                    losses, pde_scale=pde_scale, lambda_bc_eff=lbc_eff
                )
                total_loss.backward()

                # Gradient clipping — catches any remaining spike
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.adam_optimizer.step()
                phys_sched.step()

                _log_step = (
                    self._step == 1
                    or self._step % PHYSICS_LOG_FREQ == 0
                )
                if _log_step:
                    _dt = (
                        (_time.perf_counter() - _timer_ref)
                        / max(self._step - _timer_step_ref, 1) * 1000
                    )
                    _lr = self.adam_optimizer.param_groups[0]['lr']
                    _detail = (
                        self._step == 1
                        or self._step % PHYSICS_DETAIL_FREQ == 0
                    )
                    if _detail:
                        _audit = self._quick_physics_audit()
                        self._log_physics_dashboard(
                            step=self._step,
                            physics_epochs=physics_epochs,
                            elapsed=_time.time() - t1,
                            ms_per_step=_dt,
                            lr=_lr,
                            lbc=self.lambda_bc,
                            lbc_eff=lbc_eff,
                            pde_scale=pde_scale,
                            total_loss=total_loss,
                            losses=losses,
                            div_rms=_audit['div_l2'],
                            u_max=_audit['u_max'],
                            p_range=_audit['p_range'],
                            audit=_audit,
                        )
                    else:
                        self._log_physics_compact(
                            step=self._step,
                            physics_epochs=physics_epochs,
                            elapsed=_time.time() - t1,
                            ms_per_step=_dt,
                            lr=_lr,
                            lbc=self.lambda_bc,
                            lbc_eff=lbc_eff,
                            pde_scale=pde_scale,
                            total_loss=total_loss,
                            losses=losses,
                        )
                    _timer_ref      = _time.perf_counter()
                    _timer_step_ref = self._step

                # History
                self.loss_history['total'].append(total_loss.item())
                self.loss_history['pde'].append(losses['pde_raw'].item())
                self.loss_history['bc'].append(losses['bc_raw'].item())
                self.loss_history['radial'].append(losses['radial'].item())
                self.loss_history['smooth'].append(losses['smooth'].item())
                self.loss_history['positivity'].append(losses['positivity'].item())
                self.loss_history['p_interior'].append(losses['p_interior'].item())
                self.loss_history['p_grad'].append(losses['p_grad'].item())
                self.loss_history['vw_interior'].append(losses['vw_interior'].item())
                self.loss_history['lambda_bc'].append(self.lambda_bc)
                self.loss_history['continuity'].append(losses['continuity'].item())
                self.loss_history['momentum'].append(losses['momentum'].item())

                if epoch % 200 == 0:
                    ckpt = f"checkpoints/physics_adam_{epoch}.pth"
                    torch.save(self.model.state_dict(), ckpt)
                    logger.info(f"  Checkpoint: {ckpt}")

            phys_dur = _time.time() - t1
            logger.info(
                f"STAGE 1 COMPLETE  {phys_dur:.1f}s  "
                f"pde={losses.get('pde_raw', torch.tensor(0.0)).item():.3e}  "
                f"lambda_bc={self.lambda_bc:.4f}"
            )

        # =====================================================================
        # STAGE 2 — L-BFGS (lambda_bc frozen)
        # =====================================================================
        lambda_bc_frozen = self.lambda_bc
        logger.info("")
        logger.info(
            f"--- STAGE 2: L-BFGS  lambda_bc frozen={lambda_bc_frozen:.4f} ---"
        )

        lbfgs_opt = optim.LBFGS(
            self.model.parameters(),
            max_iter=lbfgs_epochs,
            history_size=50,
            tolerance_change=1e-16,  # extremely tight → runs all max_iter iterations
            tolerance_grad=1e-16,    # same — lets L-BFGS do full refinement
            line_search_fn='strong_wolfe',
        )

        # L-BFGS batch size: with second_order_graph=True the full momentum
        # equation graph is built, which is ~14s/iter at 10000 points (too slow).
        # 4000 interior keeps each iteration ~5s while still covering the domain.
        lbfgs_int = int(os.environ.get('LBFGS_INT', '4000'))
        lbfgs_bc  = int(os.environ.get('LBFGS_BC',  '1000'))
        logger.info(
            f"  L-BFGS batch: interior={lbfgs_int}  BC={lbfgs_bc}"
            f"  (Adam used {batch_size_int}/{batch_size_bc})"
        )
        int_f  = self.geometry.sample_interior(lbfgs_int, self.device)
        bc_f   = self.geometry.sample_walls(lbfgs_bc,     self.device)
        in_f   = self.geometry.sample_inlet(lbfgs_bc,     self.device)
        out_f  = self.geometry.sample_outlet(lbfgs_bc,    self.device)

        t2    = _time.time()
        iters = [0]

        def closure() -> torch.Tensor:
            iters[0] += 1
            lbfgs_opt.zero_grad()
            # second_order_graph=True: full momentum equation (incl. ν∇²u) is
            # differentiable here, so L-BFGS can drive momentum residual → 0.
            ls  = self._physics_loss(int_f, bc_f, in_f, out_f,
                                     second_order_graph=True)
            # use_p_grad=True: gp_int values are now clamped to [-5,5] before
            # squaring, preventing NaN under large L-BFGS steps.
            los = self._total_loss(ls, pde_scale=1.0,
                                   lambda_bc_eff=lambda_bc_frozen,
                                   use_p_grad=True)
            los.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            if iters[0] % 50 == 0 or iters[0] == 1:
                logger.info(
                    f"[L-BFGS] {iters[0]:4d}  "
                    f"tot={los.item():.3e}  "
                    f"pde={ls['pde_raw'].item():.3e}  "
                    f"bc={ls['bc_raw'].item():.3e}  "
                    f"p_int={ls['p_interior'].item():.3e}  "
                    f"t={_time.time()-t2:.1f}s"
                )
            return los

        lbfgs_opt.step(closure)
        lbfgs_dur = _time.time() - t2
        logger.info(f"STAGE 2 COMPLETE  {lbfgs_dur:.1f}s  iters={iters[0]}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), "checkpoints/pipe_flow_final.pth")
        logger.info("Final model saved: checkpoints/pipe_flow_final.pth")

        total_dur = wu_dur + phys_dur + lbfgs_dur
        logger.info(
            f"Total time: {total_dur:.1f}s  ({total_dur/60:.1f} min)"
        )
        logger.info("=" * 72)

        return self.loss_history
