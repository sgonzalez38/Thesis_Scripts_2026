import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

#Import the CNN via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)


# Import the CNN
from Models.FlowNet import FlowNetPyramidal 

#Memory configuration for PyTorch to allow dynamic GPU memory growth, preventing out-of-memory errors during training.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#------------------------------------------------------------------------------
# 1. Visualization and Training Utilities
#------------------------------------------------------------------------------

#This function plots the training loss, validation EPE, and slope ratio over epochs, saving the figure to disk. 
#It uses a logarithmic scale for the loss to better visualize improvements.
def plot_training_progress(history, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(history["train_loss"], label="Total Loss")
    ax[0].plot(history["charb"], label="Charb (Fidelity)", alpha=0.7)
    ax[0].set_title("Evolución de la Pérdida")
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(history["val_epe"], color='orange', label="Val EPE")
    ax[1].set_title("Validación EPE (Pixels)")
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(history["slope_value"], color='green')
    ax[2].axhline(y=1.0, color='r', linestyle='--') 
    ax[2].set_title("Ratio Pred/GT (Ideal=1.0)")
    ax[2].set_ylim([0.5, 1.5])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#This function visualizes the model's predictions on a fixed sample from the training set, showing the reference image, GT flow magnitude, 
#predicted flow magnitude, and their ratio to assess fidelity and scaling.

def visualize_fixed_sample(model, fixed_sample, device, pixel_size_m, save_path):
    model.eval()
    inp, flow_m = fixed_sample 
    inp_gpu = inp.unsqueeze(0).to(device)
    gt_px = (flow_m / pixel_size_m).numpy()

    with torch.no_grad():
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        preds = m(inp_gpu[:, 0:1], inp_gpu[:, 1:2])
        pred_px = preds["flow_l1"].squeeze(0).cpu().numpy()

    gt_mag = np.sqrt(gt_px[0]**2 + gt_px[1]**2)
    pred_mag = np.sqrt(pred_px[0]**2 + pred_px[1]**2)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    vmax = max(gt_mag.max(), pred_mag.max(), 0.1)

    ax[0].imshow(inp[0].numpy(), cmap='gray')
    ax[0].set_title("Ref-image (Training)")

    ax[1].imshow(gt_mag, cmap='magma', vmin=0, vmax=vmax)
    ax[1].set_title(f"GT Mag (Max: {gt_mag.max():.3f})")

    ax[2].imshow(pred_mag, cmap='magma', vmin=0, vmax=vmax)
    ax[2].set_title(f"Pred Mag (Max: {pred_mag.max():.3f})")

    ratio = pred_mag / (gt_mag + 1e-2)
    im = ax[3].imshow(ratio, cmap='RdYlGn', vmin=0, vmax=1.5)
    ax[3].set_title("Ratio Pred/GT")
    plt.colorbar(im, ax=ax[3])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#------------------------------------------------------------------------------
# 2. DATASET
#------------------------------------------------------------------------------
class CNNDataset(Dataset):
    def __init__(self, dataframe, input_dir, output_dir, photons_per_pixel=(1e5, 1e9), 
                 exposure_jitter=0.1, read_noise_std=2.0, seed=None):
        self.df = dataframe.reset_index(drop=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.eps = 1e-8
        self.photons_per_pixel = photons_per_pixel
        self.exposure_jitter = exposure_jitter
        self.read_noise_std = read_noise_std
        if seed is not None: np.random.seed(seed)

    def __len__(self): return len(self.df)

    def _add_shot_noise(self, img, photons): #This function simulates realistic sensor noise by applying Poisson noise based on the photon count, followed by optional Gaussian read noise.
        img = np.clip(img, 0.0, None)
        lam = img * photons
        noisy = np.random.poisson(lam).astype(np.float64)
        if self.read_noise_std > 0.0:
            noisy += np.random.normal(0.0, self.read_noise_std, img.shape)
        return np.clip(noisy / (photons + self.eps), 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ref = np.loadtxt(os.path.join(self.input_dir, row.I_ref)).astype(np.float32)
        samp = np.loadtxt(os.path.join(self.input_dir, row.I_samp)).astype(np.float32)
        dx = np.loadtxt(os.path.join(self.output_dir, row.d_x), dtype=np.float32)
        dy = np.loadtxt(os.path.join(self.output_dir, row.d_y), dtype=np.float32)

        p = self.photons_per_pixel
        photons = np.random.uniform(p[0], p[1]) if isinstance(p, tuple) else p
        if self.exposure_jitter > 0: 
            photons *= (1.0 + np.random.uniform(-self.exposure_jitter, self.exposure_jitter))

        #Normalization using max and min per pair to ensure that the input images are scaled to [0,1] while preserving relative contrast.
        pair_min, pair_max = min(ref.min(), samp.min()), max(ref.max(), samp.max())
        denom = (pair_max - pair_min) + self.eps
        ref, samp = (ref - pair_min)/denom, (samp - pair_min)/denom
        ref = self._add_shot_noise(ref, photons)
        samp = self._add_shot_noise(samp, photons)
        return torch.from_numpy(np.stack([ref, samp], axis=0)), \
               torch.from_numpy(np.stack([dx, dy], axis=0))

#------------------------------------------------------------------------------
# 3. Multiscale Loss Function
#------------------------------------------------------------------------------

def flow_loss_single(pred, gt, img, alpha_mag, p_mag, eps, delta,
                            lambda_slope, lambda_smooth, apply_slope):
    pred_mag = torch.sqrt((pred ** 2).sum(dim=1, keepdim=True) + eps)
    gt_mag = torch.sqrt((gt ** 2).sum(dim=1, keepdim=True) + eps)
    diff = pred - gt
    error_charb = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + eps**2) #Charbonnier error
    weight = (gt_mag + delta) ** p_mag
    weight = weight / (weight.mean() + eps)
    loss_charb = (error_charb * weight).mean()

    if apply_slope:
        slope = pred_mag.mean() / gt_mag.mean().clamp(min=1e-6)
        slope_loss = (slope - 1.0) ** 2
    else:
        slope_loss, slope = torch.tensor(0.0).to(pred.device), torch.tensor(1.0).to(pred.device) #Slope loss

    df_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    df_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    img_dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    img_dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    w_edge_x = torch.exp(-5.0 * img_dx)
    w_edge_y = torch.exp(-5.0 * img_dy)
    w_zero_x = torch.exp(-15.0 * gt_mag[:, :, :, :-1]) 
    w_zero_y = torch.exp(-15.0 * gt_mag[:, :, :-1, :])
    smooth_loss = (df_dx * w_edge_x * w_zero_x).mean() + (df_dy * w_edge_y * w_zero_y).mean() #Smooth loss

    total = alpha_mag * loss_charb + lambda_slope * slope_loss + lambda_smooth * smooth_loss
    return total, {"Charb": loss_charb, "Slope": slope_loss, "Smooth": smooth_loss, "SlopeVal": slope}

def multiscale_flow_loss(preds, gt_px, img, epoch):
    l_slope = min(5e-3, epoch / 40.0 * 5e-3)
    gt_l1, img_l1 = gt_px, img
    gt_l2 = F.interpolate(gt_px, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
    gt_l3 = F.interpolate(gt_px, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
    img_l2 = F.interpolate(img, scale_factor=0.5, mode="bilinear", align_corners=False)
    img_l3 = F.interpolate(img, scale_factor=0.25, mode="bilinear", align_corners=False)

    params = {"alpha_mag": 3.0, "p_mag": 1.5, "eps": 1e-4, "delta": 0.05}
    l3, _ = flow_loss_single(preds["flow_l3"], gt_l3, img_l3, **params, lambda_slope=0, lambda_smooth=0.1, apply_slope=False)
    l2, _ = flow_loss_single(preds["flow_l2"], gt_l2, img_l2, **params, lambda_slope=0, lambda_smooth=0.1, apply_slope=False)
    l1, stats = flow_loss_single(preds["flow_l1"], gt_l1, img_l1, **params, lambda_slope=l_slope, lambda_smooth=0.1, apply_slope=True)
    return 0.4*l3 + 0.7*l2 + 1.0*l1, stats

###############################################################################
# 4. ENTRENAMIENTO CON OPTIMIZACIÓN DE MEMORIA
###############################################################################

def train_flownet(model, train_loader, val_loader, train_ds, pixel_size_m, save_dir, epochs=300, lr=5e-4):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu" #Use GPU if available
    
    # Since we have two GPUs available, we can use DataParallel to automatically split the batch across them.
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    fixed_sample = train_ds[0] 

    history = {"train_loss": [], "val_epe": [], "charb": [], "slope_value": [], "lr": []}
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_running_loss, stats_running = 0.0, {"charb": 0.0, "slope_val": 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inp, flow_m in pbar:
            inp, flow_m = inp.to(device), flow_m.to(device)
            gt_px = flow_m / pixel_size_m
            
            preds = model(inp[:, 0:1], inp[:, 1:2])
            loss, stats = multiscale_flow_loss(preds, gt_px, inp[:, 0:1], epoch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            stats_running["charb"] += stats["Charb"].item()
            stats_running["slope_val"] += stats["SlopeVal"].item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
            # Explicit deletion of variables to free up GPU memory after each iteration, especially important when using DataParallel which can increase memory usage.
            del preds, loss, inp, flow_m

        # Validation
        model.eval()
        val_epe = 0.0
        with torch.no_grad():
            for inp, flow_m in val_loader:
                inp, flow_m = inp.to(device), flow_m.to(device)
                preds = model(inp[:, 0:1], inp[:, 1:2])
                val_epe += torch.norm(preds["flow_l1"] - (flow_m/pixel_size_m), dim=1).mean().item()
                del preds, inp, flow_m
        
        val_epe /= len(val_loader)
        n = len(train_loader)
        history["train_loss"].append(train_running_loss / n)
        history["charb"].append(stats_running["charb"] / n)
        history["slope_value"].append(stats_running["slope_val"] / n)
        history["val_epe"].append(val_epe)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if (epoch + 1) % 5 == 0:
            plot_training_progress(history, os.path.join(save_dir, "training_curves.png"))
            visualize_fixed_sample(model, fixed_sample, device, pixel_size_m, 
                                   os.path.join(save_dir, f"train_vis_ep{epoch+1}.png"))
        
        if val_epe < best_val:
            best_val = val_epe
            # Save the best model state dict, handling both DataParallel and single GPU cases to ensure compatibility when loading later.
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(save_dir, "best_model.pt"))
        
        scheduler.step(val_epe)
        torch.cuda.empty_cache() # erase cached memory after each epoch

    np.save(os.path.join(save_dir, "history.npy"), history)

if __name__ == "__main__":
    pixel_size_m = 55e-6 # Pixel size in meters (55 microns)
    model = FlowNetPyramidal(max_disp=2, use_cbam=False)
    
    # Data load
    df = pd.read_csv("No_Poisson/Dataset_1_no_Poisson.csv") #Change this to your dataset path if needed
    for col in ["I_ref", "I_samp", "d_x", "d_y"]:
        df[col] = df[col].str.replace('\\', '/', regex=False)
        df[col] = df[col].str.replace('^Inputs_noPoisson/', '', regex=True)
        df[col] = df[col].str.replace('^d_x_&_d_y_Images/', '', regex=True)

    mask = np.random.RandomState(42).rand(len(df)) < 0.7 #70% for training, 30% for validation split, using a fixed seed for reproducibility.
    train_ds = CNNDataset(df[mask], "No_Poisson/Inputs_NoPoisson", "No_Poisson/d_x_&_d_y_Images", seed=42)
    val_ds = CNNDataset(df[~mask], "No_Poisson/Inputs_NoPoisson", "No_Poisson/d_x_&_d_y_Images", seed=42)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    train_flownet(model, train_loader, val_loader, train_ds, pixel_size_m, save_dir="Checkpoints_FlowNet")