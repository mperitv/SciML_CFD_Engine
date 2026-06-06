import torch
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CFDVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        assert hasattr(self.model, 'hidden_dim') and hasattr(self.model, 'num_layers'), (
            'Model metadata missing: expected attributes hidden_dim and num_layers from networks.py.'
        )
        assert len(list(self.model.net)) == self.model.num_layers * 2 + 1, (
            'Model sequential structure does not match expected networks.py architecture.'
        )
        logger.info(f'Visualizer model check: hidden_dim={self.model.hidden_dim}, num_layers={self.model.num_layers}')
        self.model.eval() # Modeli test moduna alıyoruz (dropout vb. kapanır)

    def plot_pipe_slice(self, length: float, radius: float, save_path: str = "pipe_flow_result.png"):
        """
        Borunun ortasından (Z=0 düzlemi) X-Y kesiti alır ve X-Yönü Hızını (U) çizer.
        Beklenen sonuç: Duvarlarda hız 0 (mavi), borunun ortasında hız maksimum (kırmızı).
        """
        logger.info("Generating 2D slice visualization for the pipe flow...")
        
        # Kesit için grid (ızgara) oluşturma
        nx, ny = 200, 50
        x = np.linspace(0, length, nx)
        y = np.linspace(-radius, radius, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X) # Borunun tam ortasından (Z=0) kesiyoruz

        # PyTorch tensörüne çevirme
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
        z_flat = torch.tensor(Z.flatten(), dtype=torch.float32).unsqueeze(1)
        
        coords = torch.cat([x_flat, y_flat, z_flat], dim=1).to(self.device)

        # Modelden Hız ve Basınç tahminlerini alma
        with torch.no_grad():
            preds = self.model(coords)
            u_vel = preds[:, 0].cpu().numpy().reshape(ny, nx) # Sadece X-yönü hızı (U)

        # Matplotlib ile Çizim
        plt.figure(figsize=(10, 4))
        
        # Kontur çizimi
        contour = plt.contourf(X, Y, u_vel, levels=50, cmap='jet')
        plt.colorbar(contour, label='X-Velocity (u)')
        
        plt.title('AI-Predicted Velocity Profile Inside the Pipe (Z=0 Slice)')
        plt.xlabel('Pipe Length (X)')
        plt.ylabel('Pipe Radius (Y)')
        
        # Duvarları siyah çizgi ile belirginleştir
        plt.axhline(y=radius, color='black', linewidth=2, linestyle='--')
        plt.axhline(y=-radius, color='black', linewidth=2, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Visualization saved to {save_path}")
        plt.close()