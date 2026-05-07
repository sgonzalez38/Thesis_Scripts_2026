import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import sys

#Import the DDPM via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)


# Import the DDPM
from Models.Difusion_Model import DiffusionUNet

# ----------------------------------------------------------------------------
# 1. Inference dataset and model loading
# ----------------------------------------------------------------------------

class DiffusionInference:
    def __init__(self, model_path, device, mean, std, base_dim=64):
        self.device = device
        self.mean = mean
        self.std = std

        self.model = DiffusionUNet(
            in_channels=1,
            out_channels=1,
            base_dim=base_dim
        ).to(device)

        # Weight loading with checkpoint handling
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Cosine scheduler 
        self.T = 1000
        steps = torch.arange(self.T + 1, dtype=torch.float64)
        s = 0.008
        f = torch.cos(((steps / self.T) + s) / (1 + s) * math.pi / 2)**2
        alphas_cumprod = f / f[0]
        betas = torch.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.999)
        
        self.betas = betas.float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, condition):
        B, _, H, W = condition.shape
        # Start from pure noise
        x = torch.randn((B, 1, H, W), device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch, condition)

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            # Deterministic Step
            coeff = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            x = (1 / torch.sqrt(alpha)) * (x - coeff * noise_pred)

            # Adding noise except for the last step
            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t - 1]
                variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                std_dev = torch.sqrt(variance)
                x = x + std_dev * torch.randn_like(x)

        # Z-score denormalization
        return x * (self.std + 1e-8) + self.mean

# -----------------------------------------------------------------------------
# 2. DATASET DE TEST
# -----------------------------------------------------------------------------

class DifussionDataset(Dataset):
    def __init__(self, source, mean, std, input_dir=None, output_dir=None, csv_file=None):
        self.source = source
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.mean = mean
        self.std = std

        if source == "folder":
            self.paths = sorted([
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.endswith((".txt", ".npy"))
            ])
        elif source == "csv":
            self.df = pd.read_csv(csv_file)

    def clean_path(self, path, prefix):
        path = path.replace("\\", "/")
        if prefix in path:
            path = path.split(prefix, 1)[-1]
        return path.strip("/")

    def __len__(self):
        if self.source == "folder":
            return len(self.paths)
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.basename(path)
        # Load npy
        x_phys = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
        x_phys = x_phys.astype(np.float32)
        y_phys = np.zeros_like(x_phys)
        target_min, target_max = 0.0, x_phys.max()

        #Dynamic scaling to ensure the input is in a reasonable range for the model, especially if there are outliers.
        img_max = np.percentile(x_phys, 99.9)
        if img_max > 2.0:
            scale_factor = float(img_max)
            x_working = x_phys / scale_factor
        else:
            scale_factor = 1.0
            x_working = x_phys

        # Z-SCORE Normalization
        x_norm = (x_working - self.mean) / (self.std + 1e-8)
        tdenom = target_max - target_min
        return (
            torch.from_numpy(x_norm).unsqueeze(0).float(),
            torch.from_numpy(y_phys).unsqueeze(0).float(),
            name,
            target_min,
            tdenom,
            scale_factor
        )

# ==============================================================================
# 3. Run Inference
# ==============================================================================

def run_external_inference(model_path, test_folder, mean, std, save_base_dir="Inference_DDPM"):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #Use GPU if available, otherwise fallback to CPU
    os.makedirs(save_base_dir, exist_ok=True)
    
    sampler = DiffusionInference(model_path, device, mean, std)
    external_dataset = DifussionDataset(source="folder", mean=mean, std=std, input_dir=test_folder)
    external_loader = DataLoader(external_dataset, batch_size=1, shuffle=False)

    for x_norm, _, name, _, _, scale_factor in tqdm(external_loader):
        x_norm = x_norm.to(device)
        scale_factor = scale_factor.numpy()[0]
        
        # Prediction
        pred_phys = sampler.sample(x_norm).cpu().numpy()[0, 0]
        
        #Physical rescaling
        pred_phys = pred_phys * scale_factor
        pred_phys = np.clip(pred_phys, 1e-2, None)

        file_base = name[0].replace(".txt", "").replace(".npy", "")
        np.savetxt(os.path.join(save_base_dir, f"{file_base}_pred.txt"), pred_phys)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(x_norm.cpu()[0,0], cmap='magma'); plt.title("Input Speckle (Z-Norm)")
        plt.subplot(1, 2, 2); plt.imshow(pred_phys, cmap='magma'); plt.title("Physical Prediction")
        plt.savefig(os.path.join(save_base_dir, f"plot_{file_base}.png"))
        plt.close()

# ==============================================================================
# 4. PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    # Global parameters for normalization and model path
    GLOBAL_MEAN = 0.495132
    GLOBAL_STD = 0.477398
    MODEL_PATH = "../DDPM.pth"
    
    # 2. Inferencia sobre carpeta externa (sin GT)
    EXTERNAL_TEST_FOLDER = "../Test_Images/Generative"
    
    run_external_inference(
        model_path=MODEL_PATH, 
        test_folder=EXTERNAL_TEST_FOLDER, 
        mean=GLOBAL_MEAN, 
        std=GLOBAL_STD, 
        save_base_dir="Inference_DDPM"
    )