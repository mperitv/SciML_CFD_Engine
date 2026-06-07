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
WEIGHT_RADIAL:     float = 100.0   # radial guide: MSE(u, Poiseuille)
WEIGHT_SMOOTH:     float = 20.0    # axial smoothness: mean(|du/dx|²)
WEIGHT_POSITIVITY: float = 500.0   # no backflow: mean(relu(-u)²)

# ── Wang2021 config ───────────────────────────────────────────────────────────
LAMBDA_BC_INIT:   float = 1.0
LAMBDA_BC_MIN:    float = 0.1
LAMBDA_BC_MAX:    float = 20.0
GRAD_DENOM_FLOOR: float = 1e-6
WANG_UPDATE_FREQ: int   = 5        # every 5 steps (user spec)
WANG_ALPHA:       float = 0.1

# ── Training config ───────────────────────────────────────────────────────────
INTERIOR_RESAMPLE_FREQ: int   = 20
PDE_RAMP_STEPS:         int   = 100
PDE_RAMP_START:         float = 0.02


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
            'warmup':     [],
            'total':      [], 'pde':        [], 'bc':         [],
            'radial':     [], 'smooth':      [], 'positivity': [],
            'lambda_bc':  [], 'continuity':  [], 'momentum':   [],
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

    # =========================================================================
    # PHYSICS LOSS ASSEMBLY
    # =========================================================================

    def _physics_loss(
        self,
        int_pts:  torch.Tensor,
        bc_pts:   torch.Tensor,
        in_pts:   torch.Tensor,
        out_pts:  torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns all unweighted loss components.

        Forward passes: 4 (was 8 before refactoring)
          #1  compute_pde_residuals(int_pts)  preds_int + u_x reused below
          #2  model(bc_pts)                   wall no-slip
          #3  model(in_pts)                   inlet vel + inlet pressure (merged)
          #4  model(out_pts)                  outlet p=0

        Scaffolding (radial, smooth, pos) computed from #1 outputs — FREE.
        """
        # ── PDE — forward #1 ─────────────────────────────────────────────────
        cont, mx, my, mz, preds_int, u_x = self.physics.compute_pde_residuals(
            self.model, int_pts
        )
        l_cont  = torch.mean(cont ** 2)
        l_mom   = torch.mean(mx ** 2 + my ** 2 + mz ** 2)
        pde_raw = l_cont + l_mom / 3.0

        # ── Scaffolding (reuse #1 outputs — no extra forward pass) ───────────
        l_radial = self._compute_radial_guide_loss(preds_int, int_pts)
        l_smooth = self._compute_axial_smoothness_loss(u_x)
        l_pos    = self._compute_positivity_loss(preds_int)

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
            'pde_raw':    pde_raw,
            'bc_raw':     bc_raw,
            'radial':     l_radial,
            'smooth':     l_smooth,
            'positivity': l_pos,
            'continuity': l_cont,
            'momentum':   l_mom,
        }

    def _total_loss(
        self,
        losses: Dict[str, torch.Tensor],
        pde_scale: float = 1.0,
        lambda_bc_eff: float = 1.0,
    ) -> torch.Tensor:
        """
        total = pde_scale * pde_raw
              + lambda_bc_eff * bc_raw
              + WEIGHT_RADIAL    * radial
              + WEIGHT_SMOOTH    * smooth
              + WEIGHT_POSITIVITY * positivity

        Scaffolding weights are FIXED — not Wang-controlled.
        Wang2021 only adapts lambda_bc (BC vs NS physics balance).
        """
        return (
            pde_scale          * losses['pde_raw']
            + lambda_bc_eff    * losses['bc_raw']
            + WEIGHT_RADIAL    * losses['radial']
            + WEIGHT_SMOOTH    * losses['smooth']
            + WEIGHT_POSITIVITY * losses['positivity']
        )

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

            if epoch % 50 == 0 or epoch == 1:
                with torch.no_grad():
                    u_mae = torch.mean(
                        torch.abs(preds[:, 0:1] - self._u_exact(int_pts))
                    ).item()
                    p_mae = torch.mean(
                        torch.abs(preds[:, 3:4] - self._p_exact(int_pts))
                    ).item()
                logger.info(
                    f"[WarmUp] {epoch:4d}/{warmup_epochs}  "
                    f"loss={wu_loss.item():.3e}  "
                    f"u_MAE={u_mae:.4f}  p_MAE={p_mae:.4f}  "
                    f"t={_time.time()-t0:.1f}s"
                )

        wu_dur = _time.time() - t0
        logger.info(f"WARM-UP COMPLETE  {wu_dur:.1f}s")
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

                # Timing diagnostic every 10 steps
                if self._step % 10 == 0:
                    _dt = (
                        (_time.perf_counter() - _timer_ref)
                        / max(self._step - _timer_step_ref, 1) * 1000
                    )
                    logger.info(
                        f"[TIMER] step={self._step}  avg_dt={_dt:.0f}ms/step"
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
                self.loss_history['lambda_bc'].append(self.lambda_bc)
                self.loss_history['continuity'].append(losses['continuity'].item())
                self.loss_history['momentum'].append(losses['momentum'].item())

                if epoch % 50 == 0 or epoch == 1:
                    logger.info(
                        f"[Physics] {epoch:4d}/{physics_epochs}  "
                        f"tot={total_loss.item():.3e}  "
                        f"pde={losses['pde_raw'].item():.3e}  "
                        f"bc={losses['bc_raw'].item():.3e}  "
                        f"rad={losses['radial'].item():.3e}  "
                        f"smo={losses['smooth'].item():.3e}  "
                        f"pos={losses['positivity'].item():.3e}  "
                        f"lbc={self.lambda_bc:.3f}(eff={lbc_eff:.2f})  "
                        f"ramp={pde_scale:.3f}  "
                        f"t={_time.time()-t1:.1f}s"
                    )

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
            tolerance_change=1e-16,
            tolerance_grad=1e-16,
            line_search_fn='strong_wolfe',
        )

        int_f  = self.geometry.sample_interior(batch_size_int, self.device)
        bc_f   = self.geometry.sample_walls(batch_size_bc,    self.device)
        in_f   = self.geometry.sample_inlet(batch_size_bc,    self.device)
        out_f  = self.geometry.sample_outlet(batch_size_bc,   self.device)

        t2    = _time.time()
        iters = [0]

        def closure() -> torch.Tensor:
            iters[0] += 1
            lbfgs_opt.zero_grad()
            ls  = self._physics_loss(int_f, bc_f, in_f, out_f)
            los = self._total_loss(ls, pde_scale=1.0,
                                   lambda_bc_eff=lambda_bc_frozen)
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
                    f"rad={ls['radial'].item():.3e}  "
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
