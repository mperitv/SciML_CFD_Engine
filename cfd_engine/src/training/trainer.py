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
    Analytical Pressure Coupling + Physics PINN Trainer — 3D boru akisi cozucu.

    TEMEL PARADOKS VE COZUMU:
      Klasik warm-up: ag hizi parabol olarak ogrenirken basinci sifir ogrenir.
      Fizik acildiginda momentum_x = ... + p_x - nu*lap_u - driving_force
      denkleminde p_x=0 ile driving_force arasinda acik kalir → sonumlenme.

      COZUM — Analytical Pressure Coupling:
        Warm-up'ta hem hiz hem de basincin analitik degerleri agretilir:
          u_exact = 2 * relu(1 - (y^2+z^2)/R^2)   (Poiseuille)
          p_exact = p_coeff * (L - x)               (Hagen-Poiseuille)
          p_coeff = 8 * nu / R^2
        warmup_loss = MSE(u_pred, u_exact) + MSE(p_pred, p_exact)

    UC ASAMA:
      STAGE 0  Warm-Up     — analitik u+p kilidi (PDE yok)
      STAGE 1  Physics     — N-S + BC, PDE=10x, BC dinamik grad-norm
      STAGE 2  L-BFGS      — ince ayar, Stage 1 agirliklarini kullanir

    LOSS FORMULASYONU (Physics Fazinda):
      total = 10.0 * pde
            +  1.0 * radial
            + 10.0 * positivity
            +  1.0 * smooth
            + w_bc * lambda_bc * (wall + inlet + outlet)
      w_bc: her 20 epochta grad-norm ile otomatik guncellenir
    """

    def __init__(
        self,
        model: nn.Module,
        physics_engine,
        geometry_sampler,
        device: torch.device,
        lr: float = 1e-3,
        lambda_bc: float = 5000.0,
    ):
        self.model = model
        self.physics = physics_engine
        self.geometry = geometry_sampler
        self.device = device
        self.lambda_bc = lambda_bc
        self._init_lr = lr

        self.adam_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = None

        # Dinamik grad-norm agirlandirma (sadece BC icin; PDE 10x sabit)
        self._dyn_w_bc:     float = 1.0
        self._grad_ema_pde: float = 1.0
        self._grad_ema_bc:  float = 1.0

        self.loss_history: Dict[str, list] = {
            'warmup': [],
            'total': [], 'pde': [], 'radial': [], 'positivity': [],
            'smoothness': [], 'boundary': [], 'continuity': [], 'momentum': [],
        }

    # =========================================================================
    # ANALYTICAL SOLUTIONS
    # =========================================================================

    def _compute_exact_velocity(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Analitik Poiseuille hiz profili:
          u_exact = 2 * relu(1 - (y^2+z^2) / R^2)
        """
        y  = pts[:, 1:2]
        z  = pts[:, 2:3]
        R2 = self.geometry.radius ** 2
        return 2.0 * torch.relu(1.0 - (y ** 2 + z ** 2) / R2)

    def _compute_exact_pressure(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Analitik Hagen-Poiseuille basinc profili:
          p_exact = p_coeff * (L - x)
          p_coeff = 8 * nu / R^2

        Cikarim: dp/dx = nu * lap_u = (1/Re) * (-8/R^2) * u_max = -8*nu/R^2
          => p(x) = (8*nu/R^2) * (L - x)  ile p(L)=0 (outlet BC)

        Re=100, R=0.5 icin: p_coeff = 8/(100*0.25) = 0.32  (kullanici referansi)
        Re=50,  R=0.5 icin: p_coeff = 8/(50*0.25)  = 0.64

        .detach() ile koordinat gradyanlari hedef hesabina karistirilmaz.
        """
        x       = pts[:, 0:1].detach()
        p_coeff = 8.0 * self.physics.nu / (self.geometry.radius ** 2)
        return p_coeff * (self.geometry.length - x)

    # =========================================================================
    # DYNAMIC GRAD-NORM WEIGHTING (sadece BC icin)
    # =========================================================================

    def _update_grad_norms(
        self,
        pde_loss: torch.Tensor,
        bc_raw:   torch.Tensor,
        ema_alpha: float = 0.9,
    ) -> float:
        """
        BC bileseninin dinamik agirligini hesaplar.
        Hedef: PDE ve BC'nin efektif gradyan normlari esit olsun.

        Son 2D parametre matrisini proxy olarak kullanir (hesap tasarrufu).
        Sonuc w_bc [0.1, 20.0] araligina kirilir.
        """
        proxy = None
        for p in reversed(list(self.model.parameters())):
            if p.requires_grad and p.ndim == 2:
                proxy = p
                break

        if proxy is None:
            return self._dyn_w_bc

        try:
            g_pde = torch.autograd.grad(
                pde_loss, proxy,
                retain_graph=True, create_graph=False, allow_unused=True,
            )[0]
            g_bc = torch.autograd.grad(
                bc_raw, proxy,
                retain_graph=True, create_graph=False, allow_unused=True,
            )[0]

            g_pde_n = g_pde.detach().norm().item() if g_pde is not None else self._grad_ema_pde
            g_bc_n  = g_bc.detach().norm().item()  if g_bc  is not None else self._grad_ema_bc

        except RuntimeError as exc:
            logger.debug(f"Grad-norm skipped: {exc}")
            return self._dyn_w_bc

        self._grad_ema_pde = ema_alpha * self._grad_ema_pde + (1.0 - ema_alpha) * g_pde_n
        self._grad_ema_bc  = ema_alpha * self._grad_ema_bc  + (1.0 - ema_alpha) * g_bc_n

        # w_bc: BC'nin efektif gradyan normu PDE'ye esit olsun
        # Efektif PDE norm = 10.0 * ema_pde; Efektif BC norm = w_bc * ema_bc
        # => w_bc = 10.0 * ema_pde / ema_bc
        target = 10.0 * self._grad_ema_pde
        w_bc   = float(min(max(target / (self._grad_ema_bc + 1e-8), 0.1), 20.0))

        self._dyn_w_bc = w_bc
        return w_bc

    # =========================================================================
    # BOUNDARY CONDITION LOSSES
    # =========================================================================

    def _compute_inlet_loss(
        self, preds_inlet: torch.Tensor, inlet_pts: torch.Tensor
    ) -> torch.Tensor:
        """Poiseuille giris profili: u(r) = 2*(1-r^2/R^2), v=w=0."""
        y  = inlet_pts[:, 1:2]
        z  = inlet_pts[:, 2:3]
        R2 = self.geometry.radius ** 2
        u_target = 2.0 * torch.relu(1.0 - (y ** 2 + z ** 2) / R2)
        u_in = preds_inlet[:, 0:1]
        v_in = preds_inlet[:, 1:2]
        w_in = preds_inlet[:, 2:3]
        return torch.mean((u_in - u_target) ** 2) + 0.1 * torch.mean(v_in ** 2 + w_in ** 2)

    # =========================================================================
    # INTERIOR REGULARISATION LOSSES
    # =========================================================================

    def _compute_radial_guide_loss(
        self, preds_int: torch.Tensor, int_pts: torch.Tensor
    ) -> torch.Tensor:
        """Radyal kilavuz: ic noktalarda parabolik profili destekler."""
        y  = int_pts[:, 1:2]
        z  = int_pts[:, 2:3]
        R2 = self.geometry.radius ** 2
        u_target = 2.0 * torch.relu(1.0 - (y ** 2 + z ** 2) / R2)
        return torch.mean((preds_int[:, 0:1] - u_target) ** 2)

    def _compute_axial_smoothness_loss(self, int_pts: torch.Tensor) -> torch.Tensor:
        """Eksenel puruzsuzluk: mean((du/dx)^2). Tam gelismis akista du/dx=0."""
        pts_req = int_pts.clone().detach().requires_grad_(True)
        u_pred  = self.model(pts_req)[:, 0:1]
        try:
            grad = torch.autograd.grad(
                outputs=u_pred.sum(),
                inputs=pts_req,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                return torch.tensor(0.0, device=self.device)
            return torch.mean(grad[:, 0:1] ** 2)
        except RuntimeError as e:
            logger.warning(f"Smoothness grad failed: {e}")
            return torch.tensor(0.0, device=self.device)

    # =========================================================================
    # PHYSICS LOSS CALCULATOR
    # =========================================================================

    def _physics_loss(
        self,
        int_pts: torch.Tensor,
        bc_pts:  torch.Tensor,
        in_pts:  torch.Tensor,
        out_pts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        N-S + BC ham kayiplarini hesaplar.
          'boundary' = lambda_bc * bc_raw  (agirlikli)
          'bc_raw'   = wall + inlet + outlet  (grad-norm icin agirliksiz)
          'total' anahtari YOK — _assemble_total ile dis hesaplanir.
        """
        l_cont, l_mom = self.physics.compute_pde_loss(self.model, int_pts)
        pde_loss = l_cont + l_mom

        preds_int  = self.model(int_pts)
        l_pos      = torch.mean(torch.relu(-preds_int[:, 0:1]))
        l_radial   = self._compute_radial_guide_loss(preds_int, int_pts)
        l_smooth   = self._compute_axial_smoothness_loss(int_pts)

        l_wall = torch.mean(self.model(bc_pts)[:, 0:3] ** 2)
        l_in   = self._compute_inlet_loss(self.model(in_pts), in_pts)
        l_out  = torch.mean(self.model(out_pts)[:, 3:4] ** 2)
        bc_raw = l_wall + l_in + l_out
        l_bc   = self.lambda_bc * bc_raw

        return {
            'pde':        pde_loss,
            'continuity': l_cont,   'momentum':   l_mom,
            'radial':     l_radial, 'positivity': l_pos,
            'smoothness': l_smooth,
            'boundary':   l_bc,     'bc_raw':     bc_raw,
        }

    def _assemble_total(
        self,
        losses: Dict[str, torch.Tensor],
        w_bc: float,
    ) -> torch.Tensor:
        """
        Kayiplari birlestirerek total_loss uretir.
          PDE    : 10.0x sabit (fizik gucu yukseltildi)
          BC     : w_bc dinamik (grad-norm ile otomatik denge)
          Diger  : sabit katsayilar
        """
        return (
            10.0 * losses['pde']
            +  1.0 * losses['radial']
            + 10.0 * losses['positivity']
            +  1.0 * losses['smoothness']
            + w_bc * losses['boundary']
        )

    # =========================================================================
    # MAIN TRAINING ENTRY POINT
    # =========================================================================

    def train(
        self,
        adam_epochs:    int = 2000,
        lbfgs_epochs:   int = 1000,
        batch_size_int: int = 4000,
        batch_size_bc:  int = 800,
        warmup_epochs:  int = 500,
    ) -> Dict[str, list]:
        """
        Uc asamali egitim: Analytical Warm-Up -> Physics Adam -> L-BFGS

        Args:
            adam_epochs:    Toplam Adam epoch sayisi (warm-up dahil)
            lbfgs_epochs:   L-BFGS maksimum iterasyon sayisi
            batch_size_int: Ic nokta batch boyutu
            batch_size_bc:  Sinir nokta batch boyutu
            warmup_epochs:  Analitik pre-training epoch sayisi (PDE YOK)

        Returns:
            loss_history sozlugu
        """
        self.model.train()
        physics_epochs = max(0, adam_epochs - warmup_epochs)

        p_coeff = 8.0 * self.physics.nu / (self.geometry.radius ** 2)
        logger.info("=" * 70)
        logger.info("ANALYTICAL PRESSURE COUPLING PIPELINE STARTED")
        logger.info(
            f"Warm-Up: {warmup_epochs} epochs | "
            f"Physics Adam: {physics_epochs} epochs | "
            f"L-BFGS: {lbfgs_epochs} iters"
        )
        logger.info(
            f"Re={1.0/self.physics.nu:.0f} | "
            f"p_coeff={p_coeff:.4f} (p_exact=p_coeff*(L-x)) | "
            f"lambda_bc={self.lambda_bc:.1f} | PDE=10x"
        )
        logger.info("=" * 70)

        # =====================================================================
        # STAGE 0: WARM-UP — analitik u+p kilidi, PDE yok
        # =====================================================================
        logger.info("--- STAGE 0: Analytical Warm-Up (u_exact + p_exact coupling) ---")

        warmup_scheduler = CosineAnnealingLR(
            self.adam_optimizer, T_max=max(warmup_epochs, 1), eta_min=1e-5
        )
        t0 = time.time()

        for epoch in range(1, warmup_epochs + 1):
            self.adam_optimizer.zero_grad()

            int_pts = self.geometry.sample_interior(batch_size_int, self.device)

            preds  = self.model(int_pts)
            u_pred = preds[:, 0:1]
            p_pred = preds[:, 3:4]

            u_exact = self._compute_exact_velocity(int_pts)
            p_exact = self._compute_exact_pressure(int_pts)

            # Analytical Pressure Coupling: tam olarak loss_u + loss_p
            loss_u      = torch.mean((u_pred - u_exact) ** 2)
            loss_p      = torch.mean((p_pred - p_exact) ** 2)
            warmup_loss = loss_u + loss_p

            warmup_loss.backward()
            self.adam_optimizer.step()
            warmup_scheduler.step()

            self.loss_history['warmup'].append(warmup_loss.item())

            if epoch % 50 == 0 or epoch == 1:
                with torch.no_grad():
                    u_mae = torch.mean(torch.abs(u_pred - u_exact)).item()
                    p_mae = torch.mean(torch.abs(p_pred - p_exact)).item()
                elapsed = time.time() - t0
                logger.info(
                    f"[Warm-Up] Epoch {epoch:4d}/{warmup_epochs} | "
                    f"Loss: {warmup_loss.item():.3e} | "
                    f"u_MAE={u_mae:.4f}  p_MAE={p_mae:.4f} | "
                    f"LR: {warmup_scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

        warmup_dur = time.time() - t0

        # Warm-up sonu dogrulama (u ve p)
        with torch.no_grad():
            val_pts    = self.geometry.sample_interior(2000, self.device)
            val_preds  = self.model(val_pts)
            val_u_mae  = torch.mean(torch.abs(
                val_preds[:, 0:1] - self._compute_exact_velocity(val_pts)
            )).item()
            val_p_mae  = torch.mean(torch.abs(
                val_preds[:, 3:4] - self._compute_exact_pressure(val_pts)
            )).item()

        logger.info(
            f"WARM-UP COMPLETE in {warmup_dur:.1f}s | "
            f"Validation u_MAE={val_u_mae:.5f}  p_MAE={val_p_mae:.5f}"
        )
        os.makedirs("checkpoints", exist_ok=True)
        wu_ckpt = "checkpoints/warmup_pretrained.pth"
        torch.save(self.model.state_dict(), wu_ckpt)
        logger.info(f"Warm-up checkpoint saved: {wu_ckpt}")

        # =====================================================================
        # STAGE 1: PHYSICS ADAM — N-S + BC, PDE=10x, BC grad-norm dinamik
        # =====================================================================
        phys_dur   = 0.0
        total_loss = torch.tensor(0.0, device=self.device)

        if physics_epochs > 0:
            logger.info("")
            logger.info(
                "--- STAGE 1: Physics Training "
                "(N-S PDE=10x + BC lambda_bc=5000 + Grad-Norm) ---"
            )

            for pg in self.adam_optimizer.param_groups:
                pg['lr'] = self._init_lr
            physics_scheduler = CosineAnnealingLR(
                self.adam_optimizer, T_max=physics_epochs, eta_min=1e-6
            )
            self.scheduler = physics_scheduler

            t1 = time.time()

            for epoch in range(1, physics_epochs + 1):
                self.adam_optimizer.zero_grad()

                int_pts = self.geometry.sample_interior(batch_size_int, self.device)
                bc_pts  = self.geometry.sample_walls(batch_size_bc, self.device)
                in_pts  = self.geometry.sample_inlet(batch_size_bc, self.device)
                out_pts = self.geometry.sample_outlet(batch_size_bc, self.device)

                losses = self._physics_loss(int_pts, bc_pts, in_pts, out_pts)

                # Grad-norm ile BC agirligini guncelle (her 20 epochta)
                if epoch % 20 == 0 or epoch == 1:
                    self._update_grad_norms(losses['pde'], losses['bc_raw'])

                total_loss = self._assemble_total(losses, self._dyn_w_bc)

                total_loss.backward()
                self.adam_optimizer.step()
                physics_scheduler.step()

                for key in ('pde', 'radial', 'positivity', 'smoothness',
                            'boundary', 'continuity', 'momentum'):
                    self.loss_history[key].append(losses[key].item())
                self.loss_history['total'].append(total_loss.item())

                if epoch % 50 == 0 or epoch == 1:
                    elapsed = time.time() - t1
                    logger.info(
                        f"[Physics] Epoch {epoch:4d}/{physics_epochs} | "
                        f"Tot: {total_loss.item():.2e} | "
                        f"PDE: {losses['pde'].item():.2e} | "
                        f"BC: {losses['boundary'].item():.2e} | "
                        f"w_bc={self._dyn_w_bc:.2f} | "
                        f"LR: {physics_scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )

                if epoch % 200 == 0:
                    ckpt = f"checkpoints/physics_adam_{epoch}.pth"
                    torch.save(self.model.state_dict(), ckpt)
                    logger.info(f"  Checkpoint: {ckpt}")

            phys_dur = time.time() - t1
            logger.info(
                f"STAGE 1 COMPLETE in {phys_dur:.1f}s | "
                f"Final Loss: {total_loss.item():.2e} | "
                f"Final w_bc={self._dyn_w_bc:.3f}"
            )

        # =====================================================================
        # STAGE 2: L-BFGS — Stage 1 dinamik agirliklarini kullanir
        # =====================================================================
        logger.info("")
        logger.info("--- STAGE 2: L-BFGS Fine-Tuning ---")
        logger.info(
            f"Max iters: {lbfgs_epochs} | tol=1e-16 | strong_wolfe | "
            f"w_bc={self._dyn_w_bc:.3f}"
        )

        lbfgs_opt = optim.LBFGS(
            self.model.parameters(),
            max_iter=lbfgs_epochs,
            history_size=50,
            tolerance_change=1e-16,
            tolerance_grad=1e-16,
            line_search_fn='strong_wolfe',
        )

        int_pts_b  = self.geometry.sample_interior(batch_size_int, self.device)
        bc_pts_b   = self.geometry.sample_walls(batch_size_bc, self.device)
        in_pts_b   = self.geometry.sample_inlet(batch_size_bc, self.device)
        out_pts_b  = self.geometry.sample_outlet(batch_size_bc, self.device)

        t2          = time.time()
        iters       = [0]
        final_w_bc  = self._dyn_w_bc  # Stage 1 sonu degerini kilitle

        def closure() -> torch.Tensor:
            iters[0] += 1
            lbfgs_opt.zero_grad()
            losses = self._physics_loss(int_pts_b, bc_pts_b, in_pts_b, out_pts_b)
            loss   = self._assemble_total(losses, final_w_bc)
            loss.backward()
            if iters[0] % 50 == 0 or iters[0] == 1:
                elapsed = time.time() - t2
                logger.info(
                    f"[L-BFGS] Iter {iters[0]:4d} | "
                    f"Tot: {loss.item():.2e} | "
                    f"PDE: {losses['pde'].item():.2e} | "
                    f"BC: {losses['boundary'].item():.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
            return loss

        lbfgs_opt.step(closure)
        lbfgs_dur = time.time() - t2
        logger.info(f"STAGE 2 COMPLETE in {lbfgs_dur:.1f}s | Total iters: {iters[0]}")

        os.makedirs("checkpoints", exist_ok=True)
        final_path = "checkpoints/pipe_flow_final.pth"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Final model saved: {final_path}")

        total_dur = warmup_dur + phys_dur + lbfgs_dur
        logger.info(f"Total training time: {total_dur:.1f}s ({total_dur / 60:.1f}min)")
        logger.info("=" * 70)

        return self.loss_history
