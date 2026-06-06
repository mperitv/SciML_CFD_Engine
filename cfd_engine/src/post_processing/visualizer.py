from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.models.networks import PINN3DEngine

logger = logging.getLogger(__name__)

class CFDVisualizer:
    def __init__(self, model: nn.Module | None = None, device: torch.device | None = None, checkpoint_dir: str = "checkpoints"):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = model if model is not None else self._build_default_model()
        self.checkpoint_path: Path | None = None

        self._validate_model()
        self.model.eval()

    def _build_default_model(self) -> nn.Module:
        logger.info("Building default PINN3DEngine for visualization.")
        return PINN3DEngine(hidden_dim=256, num_layers=6, length=3.0, radius=0.5).to(self.device)

    def _validate_model(self) -> None:
        if not hasattr(self.model, "hidden_dim") or not hasattr(self.model, "num_layers"):
            raise TypeError("Model metadata missing: expected attributes hidden_dim and num_layers from networks.py.")

        if not hasattr(self.model, "net") or len(list(self.model.net)) != self.model.num_layers * 2 + 1:
            raise TypeError("Model sequential structure does not match expected networks.py architecture.")

        logger.info(f"Visualizer model check: hidden_dim={self.model.hidden_dim}, num_layers={self.model.num_layers}")

    def _find_checkpoint(self) -> Path | None:
        if not self.checkpoint_dir.exists():
            logger.warning(f"Checkpoint directory not found: {self.checkpoint_dir}")
            return None

        final_path = self.checkpoint_dir / "pipe_flow_final.pth"
        if final_path.exists():
            return final_path

        auto_ckpts = sorted(
            self.checkpoint_dir.glob("auto_ckpt_*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return auto_ckpts[0] if auto_ckpts else None

    def load_checkpoint(self, checkpoint_path: str | None = None) -> Path:
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        else:
            path = self._find_checkpoint() or Path()

        if not path.exists():
            raise FileNotFoundError(
                f"No checkpoint found. Checked: {self.checkpoint_dir.resolve()}\n"
                "Expected pipe_flow_final.pth or auto_ckpt_*.pth"
            )

        logger.info(f"Loading checkpoint from: {path}")
        self.checkpoint_path = path

        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and any(key.endswith("state_dict") for key in checkpoint):
                state_key = next(key for key in checkpoint if key.endswith("state_dict"))
                state_dict = checkpoint[state_key]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
        except RuntimeError as ex:
            raise RuntimeError(
                f"Model state dict mismatch while loading {path}: {ex}"
            )
        except Exception as ex:
            raise RuntimeError(f"Failed to load checkpoint {path}: {ex}")

        return path

    def plot_pipe_slice(self, length: float = 3.0, radius: float = 0.5, save_path: str = "pipe_flow_result.png", checkpoint_path: str | None = None):
        checkpoint = self.load_checkpoint(checkpoint_path)

        logger.info("Generating 2D slice visualization for the pipe flow...")
        nx, ny = 200, 50
        x = np.linspace(0, length, nx)
        y = np.linspace(-radius, radius, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
        z_flat = torch.tensor(Z.flatten(), dtype=torch.float32).unsqueeze(1)
        coords = torch.cat([x_flat, y_flat, z_flat], dim=1).to(self.device)

        with torch.no_grad():
            preds = self.model(coords)
            u_vel = preds[:, 0].cpu().numpy().reshape(ny, nx)

        u_max = float(np.max(u_vel))
        u_min = float(np.min(u_vel))
        print(f"Loaded checkpoint: {checkpoint}")
        print(f"Predicted u_min={u_min:.6f}, u_max={u_max:.6f}")

        plt.figure(figsize=(10, 4))
        contour = plt.contourf(X, Y, u_vel, levels=50, cmap="jet", vmin=0.0)
        cbar = plt.colorbar(contour, label="X-Velocity (u)")
        cbar.ax.set_ylim(0.0, cbar.ax.get_ylim()[1])

        plt.title("AI-Predicted Velocity Profile Inside the Pipe (Z=0 Slice)")
        plt.xlabel("Pipe Length (X)")
        plt.ylabel("Pipe Radius (Y)")
        plt.axhline(y=radius, color="black", linewidth=2, linestyle="--")
        plt.axhline(y=-radius, color="black", linewidth=2, linestyle="--")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        logger.info(f"Visualization saved to {save_path}")
        plt.show()
        return save_path
