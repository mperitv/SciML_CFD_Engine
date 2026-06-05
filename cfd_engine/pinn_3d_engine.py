import torch
import torch.nn as nn
import math
from typing import List, Tuple

class MultiScaleFourierEmbedding(nn.Module):
    """
    Makaledeki Denklem 1'i uygulayan Multi-Scale Random Fourier Features (RFF) katmanı.
    NTK spektrum modülasyonu ile yüksek frekanslı türbülans modlarına erişimi sağlar.
    """
    def __init__(self, in_features: int = 3, sigma_list: List[float] = [1.0, 10.0, 50.0], frequencies_per_scale: int = 42):
        super().__init__()
        self.in_features = in_features
        self.sigma_list = sigma_list
        self.frequencies_per_scale = frequencies_per_scale
        
        # B matrisinin oluşturulması (Frekans Projeksiyon Matrisi)
        # Her bir sigma değeri için Gaussian dağılımdan frekans örnekliyoruz
        B_components = []
        for sigma in sigma_list:
            # [in_features, frequencies_per_scale]
            b_scale = torch.randn(in_features, frequencies_per_scale) * sigma
            B_components.append(b_scale)
            
        # B matrisini birleştiriyoruz: [in_features, total_frequencies]
        B = torch.cat(B_components, dim=1)
        
        # B matrisini modelin bir state'i olarak kaydediyoruz (eğitilmeyecek, dondurulmuş)
        self.register_buffer('B', B)
        
        # Çıkış boyutu: Her frekans için sin ve cos (toplam frekans * 2)
        self.out_features = B.shape[1] * 2
        self.sqrt_D = math.sqrt(self.out_features)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        v: [Batch, 3] boyutunda 3D koordinatlar (x, y, z)
        """
        # Projeksiyon: 2 * pi * B * v
        # B matrisi [3, F], v matrisi [Batch, 3]. Çarpım: [Batch, F]
        proj = 2.0 * math.pi * torch.matmul(v, self.B)
        
        # Denklem 1: gamma(v) = 1/sqrt(D) * [sin(2*pi*B*v), cos(2*pi*B*v)]^T
        embedding = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) / self.sqrt_D
        return embedding


class PINN3DEngine(nn.Module):
    """
    3D Navier-Stokes çözümü için Multi-Scale Fourier Feature tabanlı Derin Sinir Ağı.
    """
    def __init__(self, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        
        # Embedding Katmanı (Makaledeki mimari)
        self.embedding = MultiScaleFourierEmbedding(
            in_features=3, 
            sigma_list=[1.0, 10.0, 50.0], 
            frequencies_per_scale=64 # Toplam 192 frekans -> 384 embedding boyutu
        )
        
        # MLP Katmanları
        layers = []
        in_dim = self.embedding.out_features
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
            
        # Çıkış Katmanı: u, v, w (hızlar) ve p (basınç)
        layers.append(nn.Linear(hidden_dim, 4))
        
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization ile NTK'nın başlangıç spektrumunu stabilize ediyoruz."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [Batch, 3] -> (x, y, z)
        return: [Batch, 4] -> (u, v, w, p)
        """
        x = self.embedding(coords)
        return self.net(x)


class NavierStokes3DLoss:
    """
    3D Sıkıştırılamaz Navier-Stokes denklemlerini hesaplayan Fizik Motoru.
    Otomatik türev (autograd) zincirini en verimli şekilde kurar.
    """
    def __init__(self, Re: float):
        self.Re = Re
        self.nu = 1.0 / Re

    def compute_gradients(self, outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """PyTorch autograd kullanarak kısmi türevleri hesaplar."""
        return torch.autograd.grad(
            outputs, inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]

    def evaluate(self, model: nn.Module, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fiziksel kayıpları (Continuity ve Momentum) hesaplar.
        """
        # Model tahmini
        preds = model(coords)
        u, v, w, p = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], preds[:, 3:4]

        # 1. Derece Türevler (Jacobian bileşenleri)
        # Tek seferde u'nun x,y,z'ye göre türevini alarak autograd overhead'ini düşürüyoruz.
        grad_u = self.compute_gradients(u, coords)
        u_x, u_y, u_z = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]

        grad_v = self.compute_gradients(v, coords)
        v_x, v_y, v_z = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]

        grad_w = self.compute_gradients(w, coords)
        w_x, w_y, w_z = grad_w[:, 0:1], grad_w[:, 1:2], grad_w[:, 2:3]

        grad_p = self.compute_gradients(p, coords)
        p_x, p_y, p_z = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

        # 2. Derece Türevler (Laplacian bileşenleri)
        u_xx = self.compute_gradients(u_x, coords)[:, 0:1]
        u_yy = self.compute_gradients(u_y, coords)[:, 1:2]
        u_zz = self.compute_gradients(u_z, coords)[:, 2:3]
        laplacian_u = u_xx + u_yy + u_zz

        v_xx = self.compute_gradients(v_x, coords)[:, 0:1]
        v_yy = self.compute_gradients(v_y, coords)[:, 1:2]
        v_zz = self.compute_gradients(v_z, coords)[:, 2:3]
        laplacian_v = v_xx + v_yy + v_zz

        w_xx = self.compute_gradients(w_x, coords)[:, 0:1]
        w_yy = self.compute_gradients(w_y, coords)[:, 1:2]
        w_zz = self.compute_gradients(w_z, coords)[:, 2:3]
        laplacian_w = w_xx + w_yy + w_zz

        # Kütle Korunumu (Continuity Equation)
        continuity = u_x + v_y + w_z

        # Momentum Denklemleri (Convection + Pressure Gradient - Diffusion = 0)
        momentum_x = (u * u_x + v * u_y + w * u_z) + p_x - self.nu * laplacian_u
        momentum_y = (u * v_x + v * v_y + w * v_z) + p_y - self.nu * laplacian_v
        momentum_z = (u * w_x + v * w_y + w * w_z) + p_z - self.nu * laplacian_w

        # MSE (Ortalama Kare Hata) hesaplaması
        loss_continuity = torch.mean(continuity**2)
        loss_momentum = torch.mean(momentum_x**2 + momentum_y**2 + momentum_z**2)

        return loss_continuity, loss_momentum


# ==========================================
# DOĞRULAMA VE TEST BLOĞU
# ==========================================
if __name__ == "__main__":
    # 1. Cihaz Ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENGINE INFO] Başlatılıyor... Hedef Cihaz: {device}")

    # 2. Simülasyon Parametreleri
    REYNOLDS_NUMBER = 10000.0 # Yüksek Re, makalendeki multiscale gücünü test etmek için
    BATCH_SIZE = 1000         # Collocation noktası sayısı
    
    # 3. Model ve Fizik Motoru Başlatma
    model = PINN3DEngine(hidden_dim=256, num_layers=6).to(device)
    physics_engine = NavierStokes3DLoss(Re=REYNOLDS_NUMBER)
    
    print(f"[ENGINE INFO] Model Yüklendi. Multi-Scale Embedding Aktif.")
    print(f"[ENGINE INFO] Reynolds Sayısı (Re): {REYNOLDS_NUMBER}\n")

    # 4. Rastgele 3D Collocation Noktaları (x, y, z)
    # create_graph=True gereksinimi için requires_grad=True OLMALIDIR.
    coords = torch.rand((BATCH_SIZE, 3), requires_grad=True, device=device)

    # 5. İleri Besleme ve Fiziksel Kayıp Hesaplama
    loss_cont, loss_mom = physics_engine.evaluate(model, coords)
    total_loss = loss_cont + loss_mom

    # 6. Sonuçların Konsola Yazdırılması
    print("="*50)
    print(" 3D NAVIER-STOKES FİZİK MOTORU TEST RAPORU ")
    print("="*50)
    print(f"  • Toplam Nokta Sayısı : {BATCH_SIZE}")
    print(f"  • Kütle Korunumu Kaybı: {loss_cont.item():.6e}")
    print(f"  • Momentum Kaybı      : {loss_mom.item():.6e}")
    print(f"  • Toplam PDE Rezidüel : {total_loss.item():.6e}")
    print("="*50)
    print("[BAŞARILI] Gradyan zincirleri kopmadan 2. derece uzaysal türevler hesaplandı.\n")