import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

#Import the GAN via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)


# Import the DDPM
from Models.Pix2PixHD import (
    GlobalGenerator,
    LocalEnhancer,
    MultiscaleDiscriminator,
    NLayerDiscriminator
)

# ------------------------------------------------------------
# 1. VGGLoss
# ------------------------------------------------------------

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters(): 
            p.requires_grad = False
        self.vgg = vgg
        self.selected = [3, 8, 17, 26, 35] #Same layers and weights as in Yu et al. (2024)
        self.weights = [1/32, 1/16, 1/8, 1/4, 1.0]
        
        # Registered as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # VGG normalization: (x - mu) / sigma
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)

        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        for w, idx in zip(self.weights, self.selected):
            fx = self.vgg[:idx+1](x)
            fy = self.vgg[:idx+1](y)
            loss += w * F.l1_loss(fx, fy)
        return loss

# -----------------------------------------------------------
# 2. GAN Loss
# -----------------------------------------------------------

class Pix2PixHDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_l1 = nn.L1Loss() #L1 Loss

    def _get_final_preds(self, preds):
        # If there is a single discriminador or multiscale
        if isinstance(preds[0], list):
            return [disc_features[-1] for disc_features in preds]
        else:
            return [preds[-1]]

    def gan_loss(self, preds, is_real): #This loss uses ReLU to implement the hinge loss.
        loss = 0.0
        final_preds = self._get_final_preds(preds)
        
        for pred_final in final_preds:
            if is_real:
                loss += torch.mean(F.relu(1.0 - pred_final))
            else:
                loss += torch.mean(F.relu(1.0 + pred_final))
        return loss

    def generator_gan_loss(self, preds):
        loss = 0.0
        final_preds = self._get_final_preds(preds)
        
        for pred_final in final_preds:
            loss -= torch.mean(pred_final)
        return loss
    
    def compute_l1(self, fake, real):
        return self.criterion_l1(fake, real)

# ------------------------------------------------------------
# 3. Dataset
# ------------------------------------------------------------

class GANDataset(Dataset):
    def __init__(self, dataframe, input_dir, output_dir, stats=None):
        self.df = dataframe.reset_index(drop=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stats = stats

    def __len__(self): return len(self.df)

    def clean_path(self, path, prefix):
        path = path.replace("\\", "/")
        if prefix in path: path = path.split(prefix, 1)[-1]
        return path.strip("/")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        in_name = self.clean_path(row["input"], "Inputs_GAN_2") #Change this acording to dataset structure
        out_name = self.clean_path(row["output"], "Outputs_GAN_2")
        
        inp_path = os.path.join(self.input_dir, in_name)
        out_path = os.path.join(self.output_dir, out_name)

        def load_fast(p):
            npy_p = p.replace('.txt', '.npy')
            if os.path.exists(npy_p): return np.load(npy_p).astype(np.float32)
            return np.loadtxt(p).astype(np.float32)

        x = load_fast(inp_path)
        y = load_fast(out_path)
        y = np.clip(y, 0.0, None)

        if self.stats:
            #Z-Score Normalization: (x - mu) / sigma
            x = (x - self.stats['mean_in']) / (self.stats['std_in'] + 1e-8)
            y = (y - self.stats['mean_out']) / (self.stats['std_out'] + 1e-8)
        
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)
    
def get_dataset_stats(df, input_dir, output_dir, save_path):
    #Calculates global mean and std for inputs and outputs, and saves them in a JSON file for future use. If the file already exists, it loads the stats from there.
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            return json.load(f)
    
    temp_ds = GANDataset(df, input_dir, output_dir, stats=None)
    loader = DataLoader(temp_ds, batch_size=32, num_workers=8)
    
    m_in, s_in, m_out, s_out = 0, 0, 0, 0 #mean and std accumulators
    n = 0
    
    for x, y in tqdm(loader):
        # x shape: [B, 1, H, W]
        m_in += x.mean().item()
        s_in += x.std().item()
        m_out += y.mean().item()
        s_out += y.std().item()
        n += 1
        
    stats = {
        'mean_in': m_in / n, 'std_in': s_in / n,
        'mean_out': m_out / n, 'std_out': s_out / n
    }
    
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    return stats
#Plotting functions for training history and sample outputs during training. These functions save the plots in the specified directories for later analysis.
def plot_history(history, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f)
    for key in history["train"].keys():
        plt.figure()
        plt.plot(history["train"][key], label=f"Train {key}")
        if key in history["val"]: plt.plot(history["val"][key], label=f"Val {key}")
        plt.yscale("log"); plt.legend(); plt.title(f"{title} - {key}")
        plt.savefig(os.path.join(save_dir, f"loss_{key}.png")); plt.close()

def save_sample_plots(G, samples, epoch, save_dir, stage):
    G.eval(); os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (x, y) in enumerate(samples):
            fake = G(x)
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(x[0,0].cpu(), cmap="gray"); axs[0].set_title("Input")
            axs[1].imshow(y[0,0].cpu(), cmap="gray"); axs[1].set_title("Target")
            axs[2].imshow(fake[0,0].cpu(), cmap="gray"); axs[2].set_title("Fake")
            plt.savefig(os.path.join(save_dir, f"{stage}_ep{epoch:03d}_{i}.png")); plt.close()

# ------------------------------------------------------------
# 4. Stage 1 training (Global)
# ------------------------------------------------------------

def train_stage1(train_loader, val_loader, device, epochs=50, save_dir="stage1"):
    G1 = GlobalGenerator(1, 1, n_downsampling=4, n_blocks=9).to(device) 
    D = NLayerDiscriminator(in_channels=2).to(device)
    opt_G = torch.optim.Adam(G1.parameters(), 2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    loss_fn = Pix2PixHDLoss()
    
    history = {"train": {"L1": []}, "val": {"L1": []}}
    fixed_samples = [(F.interpolate(x.to(device), 128), F.interpolate(y.to(device), 128)) for x, y in val_loader][:4]

    for epoch in range(epochs):
        # Training:
        G1.train(); D.train(); l1_acc, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Stage1 Ep {epoch+1}")
        for x, y in pbar:
            x, y = F.interpolate(x.to(device), 128), F.interpolate(y.to(device), 128)
            
            # Discriminator 
            with torch.no_grad(): fake = G1(x)
            d_loss = 0.5 * (loss_fn.gan_loss(D(torch.cat([x, y], 1)), True) + 
                            loss_fn.gan_loss(D(torch.cat([x, fake], 1)), False))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Generator
            fake = G1(x)
            g_l1 = loss_fn.compute_l1(fake, y)
            g_loss = loss_fn.gan_loss(D(torch.cat([x, fake], 1)), True) + 50.0 * g_l1
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            
            l1_acc += g_l1.item(); n += 1
            pbar.set_postfix({"L1": f"{l1_acc/n:.2e}"})
        
        # Validation:
        G1.eval()
        v_l1_acc = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = F.interpolate(vx.to(device), 128), F.interpolate(vy.to(device), 128)
                fake_val = G1(vx)
                v_l1_acc += loss_fn.compute_l1(fake_val, vy).item()
                
        v_l1_avg = v_l1_acc / len(val_loader)
        
        # Save in history
        history["train"]["L1"].append(l1_acc/n)
        history["val"]["L1"].append(v_l1_avg)
        
        # Logs and plots
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_history(history, save_dir, "Stage1")
            save_sample_plots(G1, fixed_samples, epoch+1, f"{save_dir}/samples", "stage1")
            torch.save(G1.state_dict(), f"{save_dir}/G1_latest.pth")
            
    return G1

# ------------------------------------------------------------
# 4. Stage 2 training (GAN + VGG)
# ------------------------------------------------------------

def train_stage2(train_loader, val_loader, device, G1, epochs=100, save_dir="stage2"):
    G2 = LocalEnhancer(1, 1, global_generator=G1).to(device)
    D = MultiscaleDiscriminator(in_channels=2, n_discriminators=2).to(device)
    
    # TTUR: Discriminator learns 4 times faster than the generator.
    opt_G = torch.optim.Adam(G2.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.999))
    
    vgg_fn = VGGLoss(device) 
    loss_fn = Pix2PixHDLoss()
    
    history = {"train": {"GAN": [], "VGG": [], "D": []}, "val": {"GAN": [], "VGG": []}}
    fixed_samples = [(x.to(device), y.to(device)) for x, y in val_loader][:4]

    for epoch in range(epochs):
        # Training:
        G2.train(); D.train()
        t_met = {"GAN": 0, "VGG": 0, "D": 0}
        n = 0
        pbar = tqdm(train_loader, desc=f"Stage2 [GAN+VGG] Ep {epoch+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Discriminator
            fake = G2(x).detach()
            d_loss = 0.5 * (loss_fn.gan_loss(D(torch.cat([x, y], 1)), True) + 
                            loss_fn.gan_loss(D(torch.cat([x, fake], 1)), False))
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Generator
            fake = G2(x)
            g_gan = loss_fn.generator_gan_loss(D(torch.cat([x, fake], 1)))
            g_vgg = vgg_fn(fake, y)
            g_tot = g_gan + 10.0 * g_vgg # 10 weight for VGG loss, can be tuned 
            
            opt_G.zero_grad(); g_tot.backward(); opt_G.step()
            
            t_met["GAN"] += g_gan.item(); t_met["VGG"] += g_vgg.item(); t_met["D"] += d_loss.item(); n += 1
            pbar.set_postfix({"VGG": f"{t_met['VGG']/n:.2e}", "D": f"{d_loss.item():.2e}"})

        # Validation:
        G2.eval()
        v_met = {"GAN": 0.0, "VGG": 0.0}
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                fake_val = G2(vx)
                
                # Validation loss
                v_gan = loss_fn.generator_gan_loss(D(torch.cat([vx, fake_val], 1)))
                v_vgg = vgg_fn(fake_val, vy)
                
                v_met["GAN"] += v_gan.item()
                v_met["VGG"] += v_vgg.item()
                
        # Average metrics
        for k in ["GAN", "VGG", "D"]: 
            history["train"][k].append(t_met[k]/n)
            
        history["val"]["GAN"].append(v_met["GAN"] / len(val_loader))
        history["val"]["VGG"].append(v_met["VGG"] / len(val_loader))

        # Logs y plots
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_history(history, save_dir, "Stage2")
            save_sample_plots(G2, fixed_samples, epoch+1, f"{save_dir}/samples", "stage2")
            torch.save(G2.state_dict(), f"{save_dir}/G2_latest.pth")
            
    return G2

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Use GPU if available, otherwise fallback to CPU.
    idx = 12 #Just for saving checkpoints in different folders, can be changed as needed.
    df = pd.read_csv("Dataset_GAN_2.csv") #Change this path according to your dataset.
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    stats_file = f"checkpoints_HD_V{idx}/dataset_stats.json" #Path to save the dataset statistics, ensuring that it is stored in the same directory as the checkpoints for consistency and easy access during training.
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    global_stats = get_dataset_stats(train_df, "Inputs_GAN_2", "Outputs_GAN_2", stats_file)

    train_ds = GANDataset(train_df, "Inputs_GAN_2", "Outputs_GAN_2", stats=global_stats)
    val_ds = GANDataset(val_df, "Inputs_GAN_2", "Outputs_GAN_2", stats=global_stats)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=8)

    G1 = train_stage1(train_loader, val_loader, device, epochs=50, save_dir=f"checkpoints_HD_V{idx}/stage1")
    G2 = train_stage2(train_loader, val_loader, device, G1, epochs=150, save_dir=f"checkpoints_HD_V{idx}/stage2")