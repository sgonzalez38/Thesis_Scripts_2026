import os

from scipy import stats
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

#Import the GAN via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)

from Models.Pix2PixHD import LocalEnhancer, GlobalGenerator

# -----------------------------------------------------------
# 1. Dataset test
# -----------------------------------------------------------

class Pix2PixHDTestDataset(Dataset):
    def __init__(self, source, stats, input_dir=None, output_dir=None, csv_file=None):
        self.source = source
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stats = stats

        if source == "folder":
            self.paths = sorted([
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.endswith(".txt") or f.endswith(".npy")
            ])
        elif source == "csv":
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("source must be 'folder' or 'csv'")

    def clean_path(self, path, prefix):
        path = path.replace("\\", "/")
        if prefix in path: path = path.split(prefix, 1)[-1]
        return path.strip("/")

    def _load_file(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        npy_path = path.replace('.txt', '.npy')
        if os.path.exists(npy_path):
            return np.load(npy_path).astype(np.float32)
        return np.loadtxt(path).astype(np.float32)

    def __len__(self):
        return len(self.paths) if self.source == "folder" else len(self.df)

    def __getitem__(self, idx):
        # 1. Data loader and preprocessing
        if self.source == "folder":
            path = self.paths[idx]
            name = os.path.basename(path)
            x = self._load_file(path)
            y = np.zeros_like(x) # Dummy GT para modo folder
        else:
            row = self.df.iloc[idx]
            in_name = self.clean_path(row["input"], "Inputs_GAN_2") #Change this according to your CSV paths
            out_name = self.clean_path(row["output"], "Outputs_GAN_2")
            name = os.path.basename(in_name)
            x = self._load_file(os.path.join(self.input_dir, in_name))
            y = self._load_file(os.path.join(self.output_dir, out_name))
            y = np.clip(y, 0.0, None)

        # 2. Z-Score Global Normalization
        x_norm = (x - self.stats['mean_in']) / (self.stats['std_in'] + 1e-8)
        
        return (
            torch.from_numpy(x_norm).unsqueeze(0), 
            torch.from_numpy(y).unsqueeze(0), 
            name
        )

# ------------------------------------------------------------
# 2. Visualization utilities
# ------------------------------------------------------------

def save_png_gt_pred(pred, gt, path, title=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    im0 = axs[0].imshow(gt, cmap="gray")
    axs[0].set_title("Ground Truth (Physical)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    im1 = axs[1].imshow(pred, cmap="gray")
    axs[1].set_title("Prediction (Physical)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    if title: fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------
# 3. Test Loop
# -----------------------------------------------------------

def run_test(model, loader, stats, out_dir, desc, make_scatter=False):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    all_gt, all_pred = [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, gt, name in tqdm(loader, desc=desc):
            x = x.to(device)
            fake = model(x)
            
            # 1. convert to numpy
            fake_np = fake.squeeze().cpu().numpy()
            gt_np = gt.squeeze().numpy()
            
            # 2. Denormalization
            fake_phys = (fake_np * stats['std_out']) + stats['mean_out']
            
            base_name = name[0].replace(".txt", "").replace(".npy", "")

            # 3. Save physical prediction as .txt
            np.savetxt(os.path.join(out_dir, f"{base_name}.txt"), fake_phys, fmt="%.6e")

            # 4. Save PNG visualization (only if GT is available, i.e., CSV mode)
            if gt.abs().sum() > 0: 
                save_png_gt_pred(fake_phys, gt_np, os.path.join(out_dir, f"{base_name}.png"), title=base_name)
                if make_scatter:
                    all_gt.append(gt_np.flatten())
                    all_pred.append(fake_phys.flatten())
            else: # Just prediction
                plt.imsave(os.path.join(out_dir, f"{base_name}.png"), fake_phys, cmap='gray')
# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use GPU if available, otherwise fallback to CPU 
    checkpoint_path = f"../"
    
    # 1. Load stats
    with open(os.path.join(checkpoint_path, "dataset_GAN_stats.json"), 'r') as f:
        global_stats = json.load(f)

    # 2. Global Generator
    base_g1 = GlobalGenerator(
        in_channels=1, 
        out_channels=1, 
        ngf=64, 
        n_downsampling=4, 
        n_blocks=9
    )

    # 3. Local Enhancer
    G = LocalEnhancer(
        in_channels=1,
        out_channels=1,
        ngf=32,
        n_local_enhancers=1,
        n_blocks=9,
        global_generator=base_g1
    ).to(device)

    # 4. Loading weights
    weights_path = os.path.join(checkpoint_path, "stage2", "GAN_G2.pth")
    if os.path.exists(weights_path):
        G.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
    else:
        print(f"No file found for {weights_path}")

    G.eval()

    # 5. Test
    if os.path.exists("../Test_Images/Generative"):
        ds_ext = Pix2PixHDTestDataset(
            source="folder",
            stats=global_stats,
            input_dir="../Test_Images/Generative"
        )
        loader_ext = DataLoader(ds_ext, batch_size=1, shuffle=False)

        run_test(
            G, loader_ext, global_stats,
            out_dir=f"Inference_GAN",
            desc="Test External",
            make_scatter=False
        )