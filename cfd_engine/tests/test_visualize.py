import sys
import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.networks import PINN3DEngine
from src.post_processing.visualizer import CFDVisualizer

if __name__ == "__main__":
    device = torch.device("cpu") # Görselleştirme için CPU yeterlidir
    
    # 1. Boş Modeli Oluştur
    model = PINN3DEngine(hidden_dim=256, num_layers=6).to(device)
    
    # 2. Eğittiğimiz Ağırlıkları Yükle
    model_path = "checkpoints/pipe_flow_model.pth"
    if not os.path.exists(model_path):
        logging.error(f"Model bulunamadı! Lütfen önce main.py'nin eğitimini bitirmesini bekleyin.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Trained model weights loaded successfully.")

    # 3. Görselleştiriciyi Çalıştır
    visualizer = CFDVisualizer(model, device)
    visualizer.plot_pipe_slice(length=3.0, radius=0.5, save_path="pipe_flow_result.png")