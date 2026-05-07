import os
import sys
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

#Import the DDPM via relative import, ensuring that the repository root is in the Python path for seamless module access.
repository_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repository_path not in sys.path:
    sys.path.append(repository_path)


# Import the DDPM
from Models.Difusion_Model import DiffusionUNet

# -----------------------------------------------------------------------------
# 1. DATASET
# -----------------------------------------------------------------------------

class DiffusionDataset(Dataset):

    def __init__(self, dataframe, input_dir, output_dir,
                 mean=0.0, std=1.0,
                 apply_poisson_to_input=False,
                 N0_range=(1e3,1e5),
                 rng=None):

        self.df = dataframe.reset_index(drop=True)
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.mean = mean
        self.std = std

        self.apply_poisson_to_input = apply_poisson_to_input
        self.N0_range = N0_range #If apply_poisson_to_input is True, this range will be used to sample the N0 parameter for the Poisson noise model, which controls the noise level added to the input data.

        self.rng = np.random.default_rng() if rng is None else rng

    def __len__(self):
        return len(self.df)

    def clean_path(self, path, prefix):
        path = path.replace("\\","/")
        if prefix in path:
            path = path.split(prefix,1)[-1]
        return path.strip("/")
    #this function samples a value of N0 from a log-uniform distribution defined by the N0_range, 
    # which is used to control the noise level when applying Poisson noise to the input data. 
    def _sample_N0(self): 
        a,b = np.log10(self.N0_range[0]), np.log10(self.N0_range[1])
        return float(10**self.rng.uniform(a,b))

    def normalize(self,x):
        return (x-self.mean)/(self.std+1e-8) #Normalize using Z-score

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        in_name  = self.clean_path(row["input"],"Inputs_GAN_2") #Adjust the prefix according to your dataset structure, this function ensures that the paths are clean and consistent regardless of how they are formatted in the CSV, which can help prevent file not found errors due to path issues.
        out_name = self.clean_path(row["output"],"Outputs_GAN_2")

        inp_path = os.path.join(self.input_dir,in_name)
        out_path = os.path.join(self.output_dir,out_name)

        if os.path.exists(inp_path.replace(".txt",".npy")):
            input_phys = np.load(inp_path.replace(".txt",".npy"))
        else:
            input_phys = np.loadtxt(inp_path)

        if os.path.exists(out_path.replace(".txt",".npy")):
            target_phys = np.load(out_path.replace(".txt",".npy"))
        else:
            target_phys = np.loadtxt(out_path)

        input_phys  = input_phys.astype(np.float32)
        target_phys = target_phys.astype(np.float32)

        if self.apply_poisson_to_input:
            N0 = self._sample_N0()
            std_noise = np.sqrt(np.abs(input_phys)*N0+1.0)/N0
            input_phys = input_phys + self.rng.normal(0,std_noise,input_phys.shape)

        input_norm  = self.normalize(input_phys)
        target_norm = self.normalize(target_phys)

        return (
            torch.from_numpy(input_norm).unsqueeze(0).float(),
            torch.from_numpy(target_norm).unsqueeze(0).float()
        )

# -----------------------------------------------------------------------------
# 2. DIFFUSION MANAGER
# -----------------------------------------------------------------------------

class DiffusionManager:

    def __init__(self,T=1000,device="cuda"):
        self.T = T
        self.device = device

        steps = torch.arange(T+1,dtype=torch.float64)
        s = 0.008
        f = torch.cos(((steps/T)+s)/(1+s)*math.pi/2)**2
        alphas_cumprod = f/f[0]
        betas = 1-(alphas_cumprod[1:]/alphas_cumprod[:-1])
        betas = torch.clip(betas,0.0001,0.999)

        self.betas = betas.float().to(device)
        self.alphas = 1-self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas,dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1-self.alphas_cumprod)

    def add_noise(self,x0,t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:,None,None,None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:,None,None,None]
        xt = sqrt_alpha_bar*x0 + sqrt_one_minus*noise

        return xt, noise

# ------------------------------------------------------------------------------
# 3. Utilities: EARLY STOPPING Y EMA
# ------------------------------------------------------------------------------

class EarlyStopping:

    def __init__(self,patience=20,min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self,val_loss):
        if val_loss < self.best_loss-self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter+=1
            print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter>=self.patience:
                self.early_stop=True
            return False

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

# ------------------------------------------------------------------------------
# 4. TRAIN LOOP
# ------------------------------------------------------------------------------

def train_diffusion(model,train_loader,val_loader,device,save_dir,epochs,lr=1e-4):

    os.makedirs(save_dir,exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    diffusion = DiffusionManager(device=device)
    stopper = EarlyStopping()
    scaler = torch.cuda.amp.GradScaler()
    
    ema = EMA(model)

    train_losses=[]
    val_losses=[]

    for epoch in range(epochs):

        model.train()
        running_train_loss=0
        pbar = tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")

        for condition,target in pbar:

            condition = condition.to(device)
            target = target.to(device)

            t = torch.randint(0,diffusion.T,(condition.shape[0],),device=device)
            noisy_target, noise = diffusion.add_noise(target,t)

            with torch.cuda.amp.autocast():
                noise_pred = model(noisy_target,t,condition)
                loss = F.mse_loss(noise_pred,noise) + 0.1*F.l1_loss(noise_pred,noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            ema.update(model)

            running_train_loss += loss.item()
            pbar.set_postfix({"loss":f"{loss.item():.5f}"})

        avg_train_loss = running_train_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        
        #Validation 
        # Save the current weights before applying EMA, so we can restore them after validation. 
        # This is important because we want to evaluate the model using the EMA weights, but we want to keep training with the original weights.
        current_weights = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        ema.apply_shadow(model)
        
        model.eval()
        running_val_loss=0
        
        # Generator for deterministic sampling during validation, ensuring that the same noise and time steps are used for each epoch, 
        # which allows for a consistent evaluation of the model's performance over time.
        val_generator = torch.Generator(device=device)
        val_generator.manual_seed(42)

        with torch.no_grad():
            for condition,target in val_loader:
                condition = condition.to(device)
                target = target.to(device)

                # Time and noise for each epoch
                t = torch.randint(0, diffusion.T, (condition.shape[0],), generator=val_generator, device=device)
                noise = torch.randn(target.shape, generator=val_generator, device=device, dtype=target.dtype)
                
                noisy_target, noise = diffusion.add_noise(target, t, noise=noise)

                # AMP for validation
                with torch.cuda.amp.autocast():
                    noise_pred = model(noisy_target,t,condition)
                    v_loss = F.mse_loss(noise_pred,noise) + 0.1*F.l1_loss(noise_pred,noise) #Validation loss, same as training loss for consistency

                running_val_loss += v_loss.item()

        avg_val_loss = running_val_loss/len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss {avg_train_loss:.6f}")
        print(f"Val   Loss {avg_val_loss:.6f}")

        is_best = stopper(avg_val_loss)

        if is_best:
            # Save the best model state dict, handling both DataParallel and single GPU cases to ensure compatibility when loading later.
            torch.save(
                {
                    "model":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "scaler":scaler.state_dict(),
                },
                os.path.join(save_dir,"best_model.pth")
            )

        # LOGS y PLOTS
        if (epoch+1)%5==0:
            pd.DataFrame(
                {"train":train_losses,"val":val_losses}
            ).to_csv(os.path.join(save_dir,"loss_log.csv"))

            plt.figure()
            plt.plot(train_losses,label="train")
            plt.plot(val_losses,label="val")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(save_dir,"loss_curve.png"))
            plt.close()

        if (epoch+1)%20==0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir,f"model_epoch_{epoch+1}.pth")
            )

        # Restore original weights after validation to continue training with the non-EMA weights, ensuring that the training process is not affected by the EMA weights which are only used for evaluation.
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = current_weights[name]
                
        if stopper.early_stop:
            print("EARLY STOPPING ALCANZADO")
            break

# ------------------------------------------------------------------------------
# 5. SAMPLING
# ------------------------------------------------------------------------------

@torch.no_grad()
def sample(model,condition,diffusion):

    device = condition.device
    x = torch.randn_like(condition)

    for t in reversed(range(diffusion.T)):
        t_tensor = torch.full((condition.shape[0],),t,device=device,dtype=torch.long)
        noise_pred = model(x,t_tensor,condition)

        alpha = diffusion.alphas[t]
        alpha_bar = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]

        if t>0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1/torch.sqrt(alpha))*(x - (beta/torch.sqrt(1-alpha_bar))*noise_pred) + torch.sqrt(beta)*noise

    return x


# ------------------------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------------------------

if __name__=="__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") #Use GPU 1 if available, otherwise fallback to CPU. Adjust the GPU index as needed based on your system configuration.

    input_dir = "Inputs_GAN_2" #Change this to your dataset path if needed
    output_dir = "Outputs_GAN_2"

    df = pd.read_csv("Dataset_GAN_2.csv") #Change this to your dataset path if needed

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))

    train_idx = idx[:int(0.8*len(df))]
    val_idx   = idx[int(0.8*len(df)):]
    
    global_mean = 0.495132 #Hardcoded global mean and std calculated from the entire dataset, used for normalization in the DiffusionDataset. This ensures that the input data is normalized consistently across all samples, which can help improve training stability and convergence.
    global_std = 0.477398
    
    train_loader = DataLoader(
        DiffusionDataset(
            df.iloc[train_idx],
            input_dir,
            output_dir,
            mean=global_mean,
            std=global_std,
            apply_poisson_to_input=False # Shot noise set to false
        ),
        batch_size=8,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        DiffusionDataset(
            df.iloc[val_idx],
            input_dir,
            output_dir,
            mean=global_mean,
            std=global_std,
            apply_poisson_to_input=False
        ),
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = DiffusionUNet(
        in_channels=1,
        out_channels=1,
        base_dim=64
    ).to(device)

    train_diffusion(
        model,
        train_loader,
        val_loader,
        device,
        save_dir="checkpoints_diffusion",
        epochs=200,
        lr=1e-4
    )