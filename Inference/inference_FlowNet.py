import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import exposure
from tqdm import tqdm
import sys

#Import the CNN via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)


# Import the CNN
from Models.FlowNet import FlowNetPyramidal as FlowNet

#----------------------------------------------------------------------------
# 1. Model loading and checkpoint handling
#----------------------------------------------------------------------------
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Dispositivo: {device}")

model = FlowNet(max_disp=2, use_cbam=False).to(device) #adjust max_disp and use_cbam according to your trained model's configuration

# Load checkpoint
ckpt_path = os.path.join(repository_path, "CNN.pt")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict):
    for key_name in ['model_state', 'state_dict', 'model_state_dict']:
        if key_name in checkpoint:
            state_dict = checkpoint[key_name]
            break
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Cleaning the 'module.' prefix if it was trained with DataParallel.
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

#------------------------------------------------------------------------------
# 2. Data Loader and preprocessing
#------------------------------------------------------------------------------
def load_and_preprocess(ref_path, samp_path):
    ref = np.loadtxt(ref_path, dtype=np.float32)
    samp = np.loadtxt(samp_path, dtype=np.float32)
    
    pair_min = float(min(ref.min(), samp.min()))
    pair_max = float(max(ref.max(), samp.max()))

    denom = (pair_max - pair_min) + 1e-8
    
    ref = (ref - pair_min) / denom
    samp = (samp - pair_min) / denom
    inp = np.stack([ref, samp], axis=0).astype(np.float32)
    return torch.from_numpy(inp).unsqueeze(0).to(device)

def safe_to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu().squeeze().numpy()
    return np.squeeze(x)

#-----------------------------------------------------------------------------
# 3. Inference
#-----------------------------------------------------------------------------
dir_test_images = os.path.join(repository_path, 'Test_Images')
test_pairs = [
    (os.path.join(dir_test_images, "Ref_Sphere.txt"), os.path.join(dir_test_images, "Samp_Sphere.txt")),
    (os.path.join(dir_test_images, "Ref_Fiber.txt"), os.path.join(dir_test_images, "Samp_Fiber.txt"))
]

output_dir = f"Inference_CNN"
os.makedirs(output_dir, exist_ok=True)

for i, (ref_path, samp_path) in enumerate(test_pairs):
    if not os.path.exists(ref_path) or not os.path.exists(samp_path):
        print(f"Skipping (file not found): {ref_path} o {samp_path}")
        continue

    x = load_and_preprocess(ref_path, samp_path)
    
    with torch.no_grad():
        out = model(x[:, 0:1], x[:, 1:2])
        pred_flow = out['flow_l1'].squeeze(0).cpu().numpy()
        heatmap_np = safe_to_numpy(out.get('heatmap_l1', None))

    d_x_pred, d_y_pred = pred_flow[0], pred_flow[1]
    
    np.savetxt(os.path.join(output_dir, f"d_x_pred_{i}.txt"), d_x_pred, fmt="%.8e")
    np.savetxt(os.path.join(output_dir, f"d_y_pred_{i}.txt"), d_y_pred, fmt="%.8e")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(d_x_pred, cmap='gray'); axes[0].set_title(f"Pred d_x {i}")
    axes[1].imshow(d_y_pred, cmap='gray'); axes[1].set_title(f"Pred d_y {i}")

    if heatmap_np is not None:
        im_h = axes[2].imshow(heatmap_np, cmap='hot'); axes[2].set_title(f"Heatmap {i}")
        plt.colorbar(im_h, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "No heatmap", ha='center', va='center')
        axes[2].set_title(f"Heatmap {i} (n/a)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediction_test_{i}.png"))
    plt.close()
