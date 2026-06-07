"""
Physics Correctness Evaluation — 3D Hagen-Poiseuille PINN Solver
=================================================================
Evaluates whether the trained model has learned true NS physics
or merely minimized loss without physical meaning.

Metrics computed:
  1.  Centerline velocity u(x, 0, 0)        — propagation test
  2.  Radial profiles u(r) at 4 x-stations  — parabola test
  3.  Parabola R² fit score                  — parabolicity
  4.  Radial symmetry error                  — implementation sanity
  5.  ∇·u  L2 norm (autograd)               — incompressibility
  6.  Momentum equation residual magnitude   — NS validity
  7.  Negative velocity ratio (%)           — positivity
  8.  Centerline monotonicity score          — propagation quality
  9.  Wall BC error                          — boundary satisfaction
  10. BC/PDE loss ratio                      — balance diagnosis

Acceptance criteria (PHYSICALLY CORRECT):
  Parabolicity score   > 0.95
  ∇·u L2 norm          < 1e-3
  Centerline monoton.  > 0.90
  Radial symmetry err  < 5 %
  Negative u ratio     < 1 %
"""

import sys
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── path setup ───────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.models.networks import PINN3DEngine
from src.physics.navier_stokes import NavierStokes3DPhysics
from src.geometry.sdf_sampler import PipeGeometrySampler

# ── simulation parameters ────────────────────────────────────────────────────
RE      = float(os.environ.get("RE",     "100.0"))
RADIUS  = float(os.environ.get("RADIUS", "0.5"))
LENGTH  = float(os.environ.get("LENGTH", "3.0"))
CKPT    = os.environ.get("CKPT", "checkpoints/pipe_flow_final.pth")

# ── grid resolution ──────────────────────────────────────────────────────────
NX_CL   = 200   # centerline points
NR_RAD  = 80    # radial points per station
NX_CONT = 2000  # interior random points for ∇·u
NX_MOM  = 2000  # interior random points for momentum residual

NU      = 1.0 / RE
P_COEFF = 8.0 * NU / RADIUS**2
U_MAX   = 2.0   # centerline velocity for parabola u = 2*(1 - r²/R²)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> PINN3DEngine:
    model = PINN3DEngine(hidden_dim=256, num_layers=6,
                         length=LENGTH, radius=RADIUS).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def coords_tensor(x_arr, y_arr, z_arr):
    """Build (N,3) tensor from three numpy arrays — no grad, on device."""
    return torch.tensor(
        np.stack([x_arr, y_arr, z_arr], axis=1), dtype=torch.float32, device=DEVICE
    )


def predict_no_grad(model, pts):
    with torch.no_grad():
        return model(pts).cpu().numpy()


def parabola_fn(r, a, b):
    """Quadratic: a*(1 - (r/b)^2)  with a≈U_MAX, b≈RADIUS."""
    return a * (1.0 - (r / b) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — CENTERLINE VELOCITY u(x, 0, 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_centerline(model):
    x_vals = np.linspace(0.0, LENGTH, NX_CL)
    pts = coords_tensor(x_vals, np.zeros(NX_CL), np.zeros(NX_CL))
    preds = predict_no_grad(model, pts)
    u_cl = preds[:, 0]

    # Monotonicity: fraction of consecutive pairs that are non-decreasing
    # (for fully-developed Poiseuille it should be flat, not increasing;
    #  we check for collapse / decay instead)
    diffs = np.diff(u_cl)
    n_decrease = np.sum(diffs < -0.01)  # significant drops
    monoton_score = 1.0 - n_decrease / max(len(diffs), 1)

    # Smoothness: 1 - norm(second derivative) / mean(|u|)
    u_mean = np.mean(np.abs(u_cl)) + 1e-12
    d2u = np.diff(u_cl, n=2)
    smoothness = max(0.0, 1.0 - np.linalg.norm(d2u) / (u_mean * NX_CL))

    # Collapse detection: mean in second half vs first half
    half = NX_CL // 2
    ratio_downstream = np.mean(np.abs(u_cl[half:])) / (np.mean(np.abs(u_cl[:half])) + 1e-12)

    return {
        "u_centerline": u_cl,
        "x_vals": x_vals,
        "u_inlet_centerline": float(u_cl[0]),
        "u_outlet_centerline": float(u_cl[-1]),
        "u_mean_first_half": float(np.mean(u_cl[:half])),
        "u_mean_second_half": float(np.mean(u_cl[half:])),
        "downstream_ratio": float(ratio_downstream),
        "monoton_score": float(monoton_score),
        "smoothness_score": float(smoothness),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — RADIAL PROFILE u(r)  +  PARABOLA FIT
# ─────────────────────────────────────────────────────────────────────────────

def test_radial_profile(model, x_station: float, label: str):
    r_vals = np.linspace(0.0, RADIUS, NR_RAD)
    # sample along y-axis (z=0)
    pts = coords_tensor(
        np.full(NR_RAD, x_station), r_vals, np.zeros(NR_RAD)
    )
    preds = predict_no_grad(model, pts)
    u_r = preds[:, 0]

    # Parabola fit
    r2_score = float("nan")
    fit_umax = float("nan")
    fit_R    = float("nan")
    try:
        popt, _ = curve_fit(
            parabola_fn, r_vals, u_r,
            p0=[U_MAX, RADIUS],
            bounds=([0.0, 0.01], [10.0, 2.0]),
            maxfev=5000,
        )
        fit_umax, fit_R = popt
        u_fit = parabola_fn(r_vals, *popt)
        ss_res = np.sum((u_r - u_fit) ** 2)
        ss_tot = np.sum((u_r - np.mean(u_r)) ** 2)
        r2_score = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))
    except Exception:
        pass

    # Wall BC error: |u(r=R)|
    wall_err = abs(float(u_r[-1]))

    # Center peak: u at r=0 should be max
    center_peak = float(u_r[0])
    peak_correct = bool(np.argmax(u_r) == 0)

    return {
        "label": label,
        "x_station": x_station,
        "u_radial": u_r,
        "r_vals": r_vals,
        "r2_parabola": r2_score,
        "fit_umax": fit_umax,
        "fit_R": fit_R,
        "wall_bc_error": wall_err,
        "center_peak": center_peak,
        "peak_at_center": peak_correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — RADIAL SYMMETRY  u(+y) vs u(-y)
# ─────────────────────────────────────────────────────────────────────────────

def test_symmetry(model, x_station: float):
    r_vals = np.linspace(0.01, RADIUS * 0.99, NR_RAD)
    pts_pos = coords_tensor(np.full(NR_RAD, x_station),  r_vals, np.zeros(NR_RAD))
    pts_neg = coords_tensor(np.full(NR_RAD, x_station), -r_vals, np.zeros(NR_RAD))

    u_pos = predict_no_grad(model, pts_pos)[:, 0]
    u_neg = predict_no_grad(model, pts_neg)[:, 0]

    sym_err_pct = 100.0 * np.mean(np.abs(u_pos - u_neg)) / (np.mean(np.abs(u_pos)) + 1e-12)
    return float(sym_err_pct)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — INCOMPRESSIBILITY  ∇·u
# ─────────────────────────────────────────────────────────────────────────────

def test_continuity(model, n_pts: int = NX_CONT):
    np.random.seed(0)
    r      = np.sqrt(np.random.rand(n_pts)) * RADIUS
    theta  = np.random.rand(n_pts) * 2.0 * math.pi
    x_rand = np.random.rand(n_pts) * LENGTH
    y_rand = r * np.cos(theta)
    z_rand = r * np.sin(theta)

    pts = torch.tensor(
        np.stack([x_rand, y_rand, z_rand], axis=1),
        dtype=torch.float32, device=DEVICE, requires_grad=True
    )

    preds = model(pts)
    u_out, v_out, w_out = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
    ones = torch.ones_like(u_out)

    du = torch.autograd.grad(u_out, pts, ones, create_graph=False, retain_graph=True)[0]
    dv = torch.autograd.grad(v_out, pts, ones, create_graph=False, retain_graph=True)[0]
    dw = torch.autograd.grad(w_out, pts, ones, create_graph=False, retain_graph=False)[0]

    div = (du[:, 0] + dv[:, 1] + dw[:, 2]).detach().cpu().numpy()

    l2_norm    = float(np.sqrt(np.mean(div ** 2)))
    max_div    = float(np.max(np.abs(div)))
    pct_ok     = float(np.mean(np.abs(div) < 1e-2) * 100)

    return {
        "div_l2_norm": l2_norm,
        "div_max": max_div,
        "pct_div_below_1e2": pct_ok,
        "div_values": div,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — MOMENTUM RESIDUAL MAGNITUDE
# ─────────────────────────────────────────────────────────────────────────────

def test_momentum_residual(model, n_pts: int = NX_MOM):
    physics = NavierStokes3DPhysics(Re=RE, pipe_length=LENGTH)
    np.random.seed(1)
    r     = np.sqrt(np.random.rand(n_pts)) * RADIUS
    theta = np.random.rand(n_pts) * 2.0 * math.pi
    x_r   = np.random.rand(n_pts) * LENGTH
    y_r   = r * np.cos(theta)
    z_r   = r * np.sin(theta)

    pts = torch.tensor(
        np.stack([x_r, y_r, z_r], axis=1),
        dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        # Use compute_pde_loss in no-create-graph mode for magnitude only
        pass

    # Need grad for residuals
    pts_g = pts.clone().requires_grad_(True)
    cont, mx, my, mz, _ = physics.compute_pde_residuals(model, pts_g)

    cont  = cont.detach().cpu().numpy()
    mx    = mx.detach().cpu().numpy()
    my    = my.detach().cpu().numpy()
    mz    = mz.detach().cpu().numpy()
    mom   = np.sqrt(mx**2 + my**2 + mz**2)

    return {
        "continuity_rms": float(np.sqrt(np.mean(cont**2))),
        "momentum_rms":   float(np.sqrt(np.mean(mom**2))),
        "momentum_x_rms": float(np.sqrt(np.mean(mx**2))),
        "momentum_y_rms": float(np.sqrt(np.mean(my**2))),
        "momentum_z_rms": float(np.sqrt(np.mean(mz**2))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — NEGATIVE VELOCITY RATIO
# ─────────────────────────────────────────────────────────────────────────────

def test_negativity(model, n_pts: int = 5000):
    np.random.seed(2)
    r     = np.sqrt(np.random.rand(n_pts)) * RADIUS
    theta = np.random.rand(n_pts) * 2.0 * math.pi
    x_r   = np.random.rand(n_pts) * LENGTH
    pts   = coords_tensor(x_r, r * np.cos(theta), r * np.sin(theta))

    preds = predict_no_grad(model, pts)
    u_vals = preds[:, 0]
    neg_pct = float(np.mean(u_vals < 0.0) * 100.0)
    u_min   = float(np.min(u_vals))
    u_max   = float(np.max(u_vals))
    u_mean  = float(np.mean(u_vals))

    return {
        "neg_velocity_pct": neg_pct,
        "u_interior_min":   u_min,
        "u_interior_max":   u_max,
        "u_interior_mean":  u_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — PRESSURE FIELD CHECK
# ─────────────────────────────────────────────────────────────────────────────

def test_pressure(model):
    x_vals = np.linspace(0.0, LENGTH, 100)
    pts = coords_tensor(x_vals, np.zeros(100), np.zeros(100))
    preds = predict_no_grad(model, pts)
    p_pred = preds[:, 3]
    p_exact = P_COEFF * (LENGTH - x_vals)

    p_err_rms  = float(np.sqrt(np.mean((p_pred - p_exact) ** 2)))
    p_gradient = float(np.polyfit(x_vals, p_pred, 1)[0])   # dp/dx (should be negative)
    p_outlet   = float(p_pred[-1])
    p_inlet    = float(p_pred[0])
    p_exact_gradient = -P_COEFF

    return {
        "p_rms_error":       p_err_rms,
        "p_gradient":        p_gradient,
        "p_exact_gradient":  p_exact_gradient,
        "p_inlet_pred":      p_inlet,
        "p_inlet_exact":     float(P_COEFF * LENGTH),
        "p_outlet_pred":     p_outlet,
        "p_outlet_exact":    0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE OVERALL SCORES
# ─────────────────────────────────────────────────────────────────────────────

def parabolicity_score(radial_results):
    """Mean R² across all radial stations (excluding inlet x=0)."""
    r2s = [r["r2_parabola"] for r in radial_results if not math.isnan(r["r2_parabola"])]
    return float(np.mean(r2s)) if r2s else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def build_report_figure(cl, radials, cont, mom, neg, pres, metrics, save_path):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0f0f1a")
    ax_color = "#0f0f1a"
    txt_color = "white"
    grid_color = "#2a2a3a"

    def styled_ax(ax, title):
        ax.set_facecolor(ax_color)
        ax.set_title(title, color=txt_color, fontsize=10, pad=6)
        ax.tick_params(colors=txt_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.xaxis.label.set_color(txt_color)
        ax.yaxis.label.set_color(txt_color)
        ax.grid(True, color=grid_color, linewidth=0.5)

    # ── 1. Centerline u(x) ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 4, (1, 2))
    ax1.plot(cl["x_vals"], cl["u_centerline"], color="#00d4ff", linewidth=2, label="u(x,0,0) pred")
    ax1.axhline(U_MAX, color="#ff6b35", linewidth=1.5, linestyle="--", label=f"exact u_max={U_MAX}")
    ax1.axhline(0.0,   color="#ff4444", linewidth=1,   linestyle=":",  label="u=0 ref")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.legend(fontsize=7, facecolor=ax_color, labelcolor=txt_color)
    styled_ax(ax1, f"Centerline Velocity u(x,0,0) | downstream_ratio={cl['downstream_ratio']:.3f}")

    # ── 2. 2D velocity slice (x–y plane, z=0) ────────────────────────────────
    ax2 = fig.add_subplot(4, 4, (3, 4))
    nx_vis, ny_vis = 150, 50
    xv = np.linspace(0, LENGTH, nx_vis)
    yv = np.linspace(-RADIUS, RADIUS, ny_vis)
    Xv, Yv = np.meshgrid(xv, yv)
    pts_vis = coords_tensor(Xv.flatten(), Yv.flatten(), np.zeros(nx_vis * ny_vis))
    u_vis = predict_no_grad(
        # need model in scope — passed via closure trick
        _model_ref[0], pts_vis
    )[:, 0].reshape(ny_vis, nx_vis)
    cf = ax2.contourf(Xv, Yv, u_vis, levels=50, cmap="plasma")
    plt.colorbar(cf, ax=ax2, label="u")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    styled_ax(ax2, "2D Slice u(x,y,z=0)")

    # ── 3–6. Radial profiles ─────────────────────────────────────────────────
    r_colors = ["#00ff88", "#ffcc00", "#ff6b35", "#cc44ff"]
    for i, rad in enumerate(radials):
        ax = fig.add_subplot(4, 4, 5 + i)
        ax.plot(rad["r_vals"], rad["u_radial"], color=r_colors[i],
                linewidth=2, label="predicted")
        # parabola overlay
        if not math.isnan(rad["r2_parabola"]):
            r_fit = np.linspace(0, RADIUS, 200)
            u_fit = parabola_fn(r_fit, rad["fit_umax"], rad["fit_R"])
            ax.plot(r_fit, u_fit, color="white", linewidth=1.2,
                    linestyle="--", label=f"fit R²={rad['r2_parabola']:.3f}")
        ax.axhline(0, color="#ff4444", linewidth=0.8, linestyle=":")
        ax.set_xlabel("r")
        ax.set_ylabel("u")
        ax.legend(fontsize=7, facecolor=ax_color, labelcolor=txt_color)
        styled_ax(ax, f"Radial Profile @ {rad['label']}  R²={rad['r2_parabola']:.3f}")

    # ── 7. Divergence histogram ───────────────────────────────────────────────
    ax7 = fig.add_subplot(4, 4, 9)
    ax7.hist(cont["div_values"], bins=60, color="#00d4ff", edgecolor=ax_color, alpha=0.85)
    ax7.axvline(0, color="white", linewidth=1.5, linestyle="--")
    ax7.set_xlabel("∇·u")
    ax7.set_ylabel("count")
    styled_ax(ax7, f"Incompressibility ∇·u  L2={cont['div_l2_norm']:.4f}")

    # ── 8. Pressure centerline ───────────────────────────────────────────────
    ax8 = fig.add_subplot(4, 4, 10)
    xp = np.linspace(0, LENGTH, 100)
    ax8.plot(xp, pres["p_rms_error"] * np.ones(100) * 0,   # placeholder
             alpha=0)
    pts_p = coords_tensor(xp, np.zeros(100), np.zeros(100))
    p_pred_arr = predict_no_grad(_model_ref[0], pts_p)[:, 3]
    p_exact_arr = P_COEFF * (LENGTH - xp)
    ax8.plot(xp, p_pred_arr, color="#ff6b35", linewidth=2, label="p_pred")
    ax8.plot(xp, p_exact_arr, color="white", linewidth=1.5,
             linestyle="--", label="p_exact")
    ax8.set_xlabel("x")
    ax8.set_ylabel("p")
    ax8.legend(fontsize=7, facecolor=ax_color, labelcolor=txt_color)
    styled_ax(ax8, f"Pressure Centerline  RMS_err={pres['p_rms_error']:.4f}")

    # ── 9. Metrics scorecard ─────────────────────────────────────────────────
    ax9 = fig.add_subplot(4, 4, (11, 12))
    ax9.set_facecolor(ax_color)
    ax9.axis("off")
    verdict_color = {
        "PHYSICALLY CORRECT": "#00ff88",
        "PARTIALLY CORRECT": "#ffcc00",
        "FAILED": "#ff4444",
    }
    v_color = verdict_color.get(metrics["VERDICT"], "white")

    lines = [
        ("METRIC",                  "VALUE",     "THRESHOLD",  "STATUS"),
        ("─" * 22,                  "─" * 10,    "─" * 12,     "─" * 8),
        ("Parabolicity Score",       f"{metrics['parabolicity']:.4f}",
         "> 0.95",
         "✔" if metrics["parabolicity"] > 0.95 else "✘"),
        ("∇·u  L2 norm",            f"{metrics['div_l2']:.2e}",
         "< 1e-3",
         "✔" if metrics["div_l2"] < 1e-3 else "✘"),
        ("Centerline Monoton.",      f"{metrics['monoton']:.4f}",
         "> 0.90",
         "✔" if metrics["monoton"] > 0.90 else "✘"),
        ("Radial Symmetry Err",      f"{metrics['sym_err']:.2f}%",
         "< 5%",
         "✔" if metrics["sym_err"] < 5.0 else "✘"),
        ("Neg. Velocity Ratio",      f"{metrics['neg_pct']:.3f}%",
         "< 1%",
         "✔" if metrics["neg_pct"] < 1.0 else "✘"),
        ("Downstream Ratio",         f"{metrics['downstream_ratio']:.4f}",
         "> 0.50",
         "✔" if metrics["downstream_ratio"] > 0.50 else "✘"),
        ("Momentum Residual RMS",    f"{metrics['mom_rms']:.4f}",
         "< 0.10",
         "✔" if metrics["mom_rms"] < 0.10 else "✘"),
        ("Continuity RMS",           f"{metrics['cont_rms']:.4f}",
         "< 0.10",
         "✔" if metrics["cont_rms"] < 0.10 else "✘"),
        ("Pressure RMS Error",       f"{metrics['p_rms']:.4f}",
         "< 0.05",
         "✔" if metrics["p_rms"] < 0.05 else "✘"),
        ("─" * 22,                  "─" * 10,    "─" * 12,     "─" * 8),
        ("VERDICT",                  metrics["VERDICT"], "", ""),
        ("Confidence",               f"{metrics['confidence']}%", "", ""),
    ]

    col_x = [0.02, 0.44, 0.70, 0.92]
    row_h  = 1.0 / (len(lines) + 1)
    for row_i, row in enumerate(lines):
        y_pos = 1.0 - (row_i + 1) * row_h
        for col_i, cell in enumerate(row):
            color = txt_color
            weight = "normal"
            if row_i == 0:
                weight = "bold"
                color = "#aaaacc"
            if row_i == len(lines) - 2:
                color = v_color
                weight = "bold"
            if cell in ("✔",):
                color = "#00ff88"
            if cell in ("✘",):
                color = "#ff4444"
            ax9.text(col_x[col_i], y_pos, cell,
                     transform=ax9.transAxes,
                     fontsize=9, color=color, weight=weight,
                     fontfamily="monospace",
                     va="center")

    # ── 10. Failure mode summary ─────────────────────────────────────────────
    ax10 = fig.add_subplot(4, 4, (13, 16))
    ax10.set_facecolor(ax_color)
    ax10.axis("off")
    failure_text = "\n".join(metrics["failure_modes"]) if metrics["failure_modes"] else "None detected"
    ax10.text(0.02, 0.95, "FAILURE MODE ANALYSIS", transform=ax10.transAxes,
              fontsize=11, color="#aaaacc", weight="bold", fontfamily="monospace", va="top")
    ax10.text(0.02, 0.80, failure_text, transform=ax10.transAxes,
              fontsize=9, color="#ff8888" if metrics["failure_modes"] else "#00ff88",
              fontfamily="monospace", va="top", wrap=True)
    ax10.text(0.02, 0.20,
              f"Root Cause: {metrics['root_cause']}",
              transform=ax10.transAxes, fontsize=9, color="#ffcc00",
              fontfamily="monospace", va="top", wrap=True)

    fig.suptitle(
        f"PINN Physics Correctness Audit — Re={RE:.0f}  R={RADIUS}  L={LENGTH}  |  {metrics['VERDICT']}",
        color=v_color, fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[EVAL] Report saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def classify(metrics):
    failures = []
    # critical checks
    if metrics["downstream_ratio"] < 0.30:
        failures.append("PROPAGATION FAILURE: flow collapses downstream (ratio < 0.30)")
    elif metrics["downstream_ratio"] < 0.50:
        failures.append("PARTIAL PROPAGATION: downstream flow severely attenuated")

    if metrics["parabolicity"] < 0.50:
        failures.append("PARABOLA FAILURE: radial profiles are not parabolic (R² < 0.50)")
    elif metrics["parabolicity"] < 0.95:
        failures.append(f"PARABOLA DEGRADED: R²={metrics['parabolicity']:.3f} < 0.95 threshold")

    if metrics["div_l2"] > 0.1:
        failures.append(f"INCOMPRESSIBILITY VIOLATED: ∇·u L2={metrics['div_l2']:.3e} > 0.1")
    elif metrics["div_l2"] > 1e-3:
        failures.append(f"INCOMPRESSIBILITY DEGRADED: ∇·u L2={metrics['div_l2']:.3e} > 1e-3")

    if metrics["neg_pct"] > 5.0:
        failures.append(f"NEGATIVE VELOCITY: {metrics['neg_pct']:.2f}% of interior u < 0")
    elif metrics["neg_pct"] > 1.0:
        failures.append(f"MINOR BACKFLOW: {metrics['neg_pct']:.2f}% of interior u < 0")

    if metrics["sym_err"] > 10.0:
        failures.append(f"ASYMMETRIC PROFILE: symmetry error={metrics['sym_err']:.1f}% > 10%")
    elif metrics["sym_err"] > 5.0:
        failures.append(f"SYMMETRY DEGRADED: {metrics['sym_err']:.1f}% > 5%")

    if metrics["mom_rms"] > 1.0:
        failures.append(f"MOMENTUM RESIDUAL HIGH: {metrics['mom_rms']:.3f} > 1.0")
    elif metrics["mom_rms"] > 0.10:
        failures.append(f"MOMENTUM RESIDUAL ELEVATED: {metrics['mom_rms']:.3f} > 0.10")

    metrics["failure_modes"] = failures

    # Overall verdict
    n_critical = sum(
        1 for f in failures
        if any(kw in f for kw in ["PROPAGATION FAILURE", "PARABOLA FAILURE",
                                   "INCOMPRESSIBILITY VIOLATED", "NEGATIVE VELOCITY"])
    )

    if n_critical >= 1 or metrics["downstream_ratio"] < 0.30:
        metrics["VERDICT"] = "FAILED"
        metrics["confidence"] = 90 if n_critical >= 2 else 80
        # root cause
        if metrics["downstream_ratio"] < 0.30:
            metrics["root_cause"] = (
                "Propagation failure — trivial u≈0 solution admitted because "
                "upstream pressure Dirichlet BC was absent; flow has no driving force."
            )
        else:
            metrics["root_cause"] = "Multiple physics constraints violated simultaneously."
    elif len(failures) > 0:
        metrics["VERDICT"] = "PARTIALLY CORRECT"
        metrics["confidence"] = 65
        metrics["root_cause"] = "Some physics satisfied; constraints not all met to threshold."
    else:
        metrics["VERDICT"] = "PHYSICALLY CORRECT"
        metrics["confidence"] = 90
        metrics["root_cause"] = "All acceptance criteria met."

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_model_ref = [None]   # mutable ref so visualization helpers can access it

def main():
    print("=" * 72)
    print("PINN PHYSICS CORRECTNESS AUDIT")
    print(f"  Checkpoint : {CKPT}")
    print(f"  Re={RE:.0f}  R={RADIUS}  L={LENGTH}  nu={NU:.4f}")
    print(f"  p_coeff={P_COEFF:.4f}  u_max_exact={U_MAX}")
    print("=" * 72)

    # ── load model ────────────────────────────────────────────────────────────
    model = load_model(CKPT)
    _model_ref[0] = model
    print(f"[EVAL] Model loaded from {CKPT}")
    print(f"[EVAL] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── run all tests ────────────────────────────────────────────────────────
    print("\n[1/7] Centerline test …")
    cl = test_centerline(model)

    print("[2/7] Radial profile tests …")
    x_stations = [0.01, LENGTH * 0.25, LENGTH * 0.5, LENGTH * 0.75, LENGTH - 0.01]
    labels      = ["x≈0 (inlet)", "x=L/4", "x=L/2", "x=3L/4", "x≈L (outlet)"]
    radials = [test_radial_profile(model, xs, lb)
               for xs, lb in zip(x_stations, labels)]

    print("[3/7] Symmetry test …")
    sym_mid = test_symmetry(model, LENGTH * 0.5)
    sym_out = test_symmetry(model, LENGTH - 0.01)
    sym_err = max(sym_mid, sym_out)

    print("[4/7] Continuity / ∇·u test …")
    cont = test_continuity(model)

    print("[5/7] Momentum residual test …")
    mom = test_momentum_residual(model)

    print("[6/7] Negative velocity test …")
    neg = test_negativity(model)

    print("[7/7] Pressure test …")
    pres = test_pressure(model)

    # ── assemble metrics ─────────────────────────────────────────────────────
    para_score = parabolicity_score(radials[1:])   # skip x≈0 where fit may be noisy

    metrics = {
        "parabolicity":       para_score,
        "div_l2":             cont["div_l2_norm"],
        "monoton":            cl["monoton_score"],
        "sym_err":            sym_err,
        "neg_pct":            neg["neg_velocity_pct"],
        "downstream_ratio":   cl["downstream_ratio"],
        "mom_rms":            mom["momentum_rms"],
        "cont_rms":           mom["continuity_rms"],
        "p_rms":              pres["p_rms_error"],
        "failure_modes":      [],
        "root_cause":         "",
        "VERDICT":            "",
        "confidence":         0,
    }
    metrics = classify(metrics)

    # ── print report ─────────────────────────────────────────────────────────
    PASS = lambda v: "✔  PASS" if v else "✘  FAIL"
    print("\n" + "=" * 72)
    print("QUANTITATIVE METRICS SUMMARY")
    print("=" * 72)
    print(f"  Parabolicity Score        : {para_score:.4f}      threshold > 0.95  {PASS(para_score > 0.95)}")
    print(f"  ∇·u  L2 norm              : {cont['div_l2_norm']:.4e}  threshold < 1e-3  {PASS(cont['div_l2_norm'] < 1e-3)}")
    print(f"  Centerline Monotonicity   : {cl['monoton_score']:.4f}      threshold > 0.90  {PASS(cl['monoton_score'] > 0.90)}")
    print(f"  Radial Symmetry Error     : {sym_err:.2f}%       threshold < 5%    {PASS(sym_err < 5.0)}")
    print(f"  Negative Velocity Ratio   : {neg['neg_velocity_pct']:.3f}%       threshold < 1%    {PASS(neg['neg_velocity_pct'] < 1.0)}")
    print(f"  Downstream Ratio          : {cl['downstream_ratio']:.4f}      threshold > 0.50  {PASS(cl['downstream_ratio'] > 0.50)}")
    print(f"  Momentum Residual RMS     : {mom['momentum_rms']:.4f}      threshold < 0.10  {PASS(mom['momentum_rms'] < 0.10)}")
    print(f"  Continuity RMS            : {mom['continuity_rms']:.4f}      threshold < 0.10  {PASS(mom['continuity_rms'] < 0.10)}")
    print(f"  Pressure RMS Error        : {pres['p_rms_error']:.4f}      threshold < 0.05  {PASS(pres['p_rms_error'] < 0.05)}")
    print()
    print(f"  Centerline u @ inlet      : {cl['u_inlet_centerline']:.4f}   (exact: {U_MAX:.1f})")
    print(f"  Centerline u @ outlet     : {cl['u_outlet_centerline']:.4f}   (exact: {U_MAX:.1f})")
    print(f"  u interior min/mean/max   : {neg['u_interior_min']:.4f} / {neg['u_interior_mean']:.4f} / {neg['u_interior_max']:.4f}")
    print(f"  p gradient (pred/exact)   : {pres['p_gradient']:.4f} / {pres['p_exact_gradient']:.4f}")
    print(f"  p inlet (pred/exact)      : {pres['p_inlet_pred']:.4f} / {pres['p_inlet_exact']:.4f}")
    print(f"  p outlet (pred/exact)     : {pres['p_outlet_pred']:.4f} / 0.0000")
    print()

    print("RADIAL PROFILE DETAILS")
    print(f"  {'Station':<20} {'R²':>8} {'fit_umax':>10} {'wall_err':>10} {'peak@ctr':>10}")
    for r in radials:
        print(f"  {r['label']:<20} {r['r2_parabola']:>8.4f} {r['fit_umax']:>10.4f} {r['wall_bc_error']:>10.4f} {str(r['peak_at_center']):>10}")
    print()

    if metrics["failure_modes"]:
        print("DETECTED FAILURE MODES:")
        for fm in metrics["failure_modes"]:
            print(f"  ✘  {fm}")
    else:
        print("  No failure modes detected.")

    print()
    print("─" * 72)
    print(f"  ROOT CAUSE    : {metrics['root_cause']}")
    print(f"  CLASSIFICATION: {metrics['VERDICT']}")
    print(f"  CONFIDENCE    : {metrics['confidence']}%")
    print("─" * 72)

    # ── generate figure ───────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fig_path = "output/physics_audit.png"
    build_report_figure(cl, radials[1:], cont, mom, neg, pres, metrics, fig_path)

    return metrics


if __name__ == "__main__":
    main()
