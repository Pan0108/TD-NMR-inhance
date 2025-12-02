# relax_denoising_transformer_cnn_1d_v16_tv_loss.py
# 描述: 终极平滑优化版本。
#       1. Loss: 引入 Total Variation (TV) Loss，强力抑制首尾震荡和整体毛刺。
#       2. Data: 修正 Shift 逻辑，尾部严格补 0 (符合物理衰减)，防止尾部上翘。
#       3. Weight: 增加了频域损失的权重，进一步确保波形一致性。

import os
import time
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ======================
# CONFIGURATION SETTINGS
# ======================
config = {
    "experiment_name": "real_data_tv_loss",
    "run_dir": "results_real_data_tv_loss",
    
    "data_paths": {
        "base_processed": r"E:\AI\PAN2025\DL_TDNMR_Inhance\data\processed_data_relaxation",
        "bandwidths": ["100khz", "200khz", "333khz", "500khz", "1000khz", "2083khz"],
    },

    "num_acquisitions": 128,
    "target_length": 4096,
    "real_data_val_split_ratio": 0.2, 
    
    "resume_training": False,
    "checkpoint_path": "results_real_data_tv_loss/best_model.pth",
    "num_epochs": 1000, 
    "minibatch_size": 32, 
    "early_stopping_patience": 50, 
    
    "learning_rate": 1e-4, 
    "optimizer_weight_decay": 1e-5,
    
    "aug_gain_range": [0.5, 2.0], 
    "aug_noise_gain_range": [0.5, 20.0], 
    "aug_shift_range": 5, 
    
    "model": {
        "base_channels": 32,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "patch_size": 1
    },
}

# ======================
# HELPER FUNCTIONS
# ======================
def process_signal_length(signal, target_length, is_train):
    current_length = signal.shape[0]
    if current_length > target_length:
        if is_train:
            start = random.randint(0, current_length - target_length)
        else:
            start = 0 
        return signal[start : start + target_length]
    elif current_length < target_length:
        pad_len = target_length - current_length
        # 保持 reflect，因为这对于卷积层处理边界最友好
        return np.pad(signal, (0, pad_len), 'reflect') 
    else:
        return signal

# ======================
# LOSS FUNCTIONS (NEW: TV Loss)
# ======================
class TVLoss(nn.Module):
    """
    全变分损失 (Total Variation Loss)
    计算相邻点之间的差值绝对值之和。最小化 TV Loss 会使曲线变平滑，
    有效消除高频毛刺和边缘的剧烈震荡。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (Batch, Length) or (Batch, 1, Length)
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # 计算 x[t+1] - x[t]
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        return torch.mean(diff)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.1):
        super().__init__()
        self.alpha = alpha # FFT Loss 权重 (调大一点，加强整体波形约束)
        self.beta = beta   # TV Loss 权重 (控制平滑度)
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.tv = TVLoss()

    def forward(self, inputs, targets):
        # 1. 时域损失 (MSE)
        time_loss = self.mse(inputs, targets)
        
        # 2. 频域损失 (L1)
        input_fft = torch.fft.rfft(inputs, dim=-1, norm='ortho')
        target_fft = torch.fft.rfft(targets, dim=-1, norm='ortho')
        input_mag = torch.abs(input_fft)
        target_mag = torch.abs(target_fft)
        freq_loss = self.l1(input_mag, target_mag)
        
        # 3. 平滑损失 (TV)
        # 我们只希望 Output 平滑，不需要计算 Target 的 TV (Target本来就是平滑的)
        tv_loss = self.tv(inputs) 
        
        return time_loss + (self.alpha * freq_loss) + (self.beta * tv_loss)

# ======================
# MODEL DEFINITION (Hybrid)
# ======================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        if self.downsample: residual = self.downsample(x)
        out += residual
        return self.relu(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x)); return x

class HybridTransformerCNN1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        base_ch = model_cfg["base_channels"]
        embed_dim = model_cfg["embed_dim"]
        patch_size = model_cfg["patch_size"]
        
        self.enc1 = nn.Sequential(ResidualBlock1D(1, base_ch), ResidualBlock1D(base_ch, base_ch))
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(ResidualBlock1D(base_ch, base_ch*2), ResidualBlock1D(base_ch*2, base_ch*2))
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = nn.Sequential(ResidualBlock1D(base_ch*2, base_ch*4), ResidualBlock1D(base_ch*4, base_ch*4))
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = nn.Sequential(ResidualBlock1D(base_ch*4, base_ch*8), ResidualBlock1D(base_ch*8, base_ch*8))
        self.pool4 = nn.MaxPool1d(2)
        
        self.seq_len = config["target_length"] // 16 
        self.to_tokens = nn.Conv1d(base_ch*8, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len // patch_size, embed_dim))
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, model_cfg["num_heads"], model_cfg["mlp_ratio"], model_cfg["dropout"]) for _ in range(model_cfg["num_layers"])])
        self.from_tokens = nn.ConvTranspose1d(embed_dim, base_ch*8, kernel_size=patch_size, stride=patch_size)

        self.upconv4 = nn.ConvTranspose1d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(ResidualBlock1D(base_ch*12, base_ch*4), ResidualBlock1D(base_ch*4, base_ch*4))
        self.upconv3 = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ResidualBlock1D(base_ch*6, base_ch*2), ResidualBlock1D(base_ch*2, base_ch*2))
        self.upconv2 = nn.ConvTranspose1d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ResidualBlock1D(base_ch*3, base_ch), ResidualBlock1D(base_ch, base_ch))
        self.upconv1 = nn.ConvTranspose1d(base_ch, base_ch, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ResidualBlock1D(base_ch*2, base_ch), ResidualBlock1D(base_ch, base_ch))
        self.final = nn.Conv1d(base_ch, 1, kernel_size=1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        x_mid = self.pool4(e4)
        tokens = rearrange(self.to_tokens(x_mid), 'b c l -> b l c')
        tokens = self.transformer(tokens + self.pos_embed)
        x_mid = self.from_tokens(rearrange(tokens, 'b l c -> b c l'))
        d4 = self.dec4(torch.cat([self.upconv4(x_mid), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final(d1).squeeze(1)

# ======================
# DATASET (Physics-Correct Shift)
# ======================
class RealAugmentedDataset(Dataset):
    def __init__(self, clean_files, noise_files, target_length=4096, is_train=True, fixed_noise_scale=None):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.target_length = target_length
        self.is_train = is_train
        self.fixed_noise_scale = fixed_noise_scale
        
        print(f"[{'Train' if is_train else 'Val'}] 正在预加载 {len(clean_files)} 个真实干净信号...")
        self.clean_cache = []
        for f in clean_files:
            try:
                sig = np.load(f).astype(np.float32)
                sig = sig / config["num_acquisitions"]
                self.clean_cache.append(sig)
            except Exception as e:
                print(f"加载失败: {f}, error: {e}")
        if not self.clean_cache: raise ValueError("未成功加载任何干净信号！")

    def __len__(self):
        return len(self.noise_files) if self.is_train else len(self.clean_cache)

    def __getitem__(self, idx):
        try:
            if self.is_train:
                idx1 = random.randint(0, len(self.clean_cache) - 1)
                idx2 = random.randint(0, len(self.clean_cache) - 1)
                sig1, sig2 = self.clean_cache[idx1], self.clean_cache[idx2]
                min_len = min(len(sig1), len(sig2))
                s1, s2 = sig1[:min_len], sig2[:min_len]
                
                alpha = random.uniform(0.1, 0.9)
                base_clean = alpha * s1 + (1 - alpha) * s2
                base_clean *= random.uniform(*config["aug_gain_range"])
                
                # [Shift Fix] 物理修正
                shift = random.randint(-config["aug_shift_range"], config["aug_shift_range"])
                if shift > 0:
                    # 向右移 (左侧空缺): 用反射填充左侧，保证 t=0 平滑
                    base_clean = np.roll(base_clean, shift)
                    base_clean[:shift] = base_clean[shift] 
                elif shift < 0:
                    # 向左移 (右侧空缺): 物理上右侧是衰减到0的，所以严格补0
                    # 这解决了尾部上翘的问题
                    base_clean = np.roll(base_clean, shift)
                    base_clean[shift:] = 0 
                
                clean_filename = "mixed_sample"
                noise_path = self.noise_files[idx]
                noise_scale = random.uniform(*config["aug_noise_gain_range"])

            else:
                base_clean = self.clean_cache[idx]
                clean_filename = Path(self.clean_files[idx]).name
                noise_idx = (idx * 997) % len(self.noise_files)
                noise_path = self.noise_files[noise_idx]
                if self.fixed_noise_scale is not None:
                    noise_scale = self.fixed_noise_scale
                else:
                    eval_levels = [1.0, 5.0, 10.0, 15.0]
                    noise_scale = eval_levels[idx % len(eval_levels)]

            target_signal = process_signal_length(base_clean, self.target_length, self.is_train)
            noise_sample = np.load(noise_path).astype(np.float32)
            noise_component = process_signal_length(noise_sample, self.target_length, self.is_train)
            noise_component *= noise_scale
            input_signal = target_signal + noise_component
            
            max_val = np.max(np.abs(input_signal))
            if max_val < 1e-6: max_val = 1.0
            
            return (torch.from_numpy((input_signal / max_val).astype(np.float32)), 
                    torch.from_numpy((target_signal / max_val).astype(np.float32)), 
                    max_val, clean_filename)
                    
        except Exception as e:
            print(f"Error: {e}. Skipping...")
            return self.__getitem__((idx + 1) % len(self))

# ======================
# RUNNERS
# ======================
def gather_file_lists(config):
    base_dir = Path(config["data_paths"]["base_processed"])
    bandwidths = config["data_paths"]["bandwidths"]
    all_clean, all_noise = [], []
    print(f"正在遍历: {bandwidths}")
    for bw in bandwidths:
        c = sorted(list((base_dir / bw / "clean").glob("*.npy")))
        n = sorted(list((base_dir / bw / "noisy").glob("*.npy")))
        all_clean.extend(c); all_noise.extend(n)
    print(f"Clean: {len(all_clean)}, Noisy: {len(all_noise)}")
    return all_clean, all_noise

def run_training(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
    
    all_clean, all_noise = gather_file_lists(config)
    random.seed(42); random.shuffle(all_clean)
    split = int(len(all_clean) * (1 - config["real_data_val_split_ratio"]))
    train_c, val_c = all_clean[:split], all_clean[split:]
    
    train_loader = DataLoader(RealAugmentedDataset(train_c, all_noise, is_train=True), 
                              batch_size=config["minibatch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(RealAugmentedDataset(val_c, all_noise, is_train=False), 
                            batch_size=config["minibatch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    model = HybridTransformerCNN1D(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["optimizer_weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 【重要】启用带 TV Loss 的 CombinedLoss
    criterion = CombinedLoss(alpha=0.2, beta=0.1).to(device)
    
    best_loss = float('inf'); patience_cnt = 0
    
    if config["resume_training"]:
        ckpt_path = Path(config["run_dir"]) / 'best_model.pth'
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            print("Resuming training...")

    for epoch in range(config["num_epochs"]):
        model.train(); train_loss = 0
        for inp, tgt, _, _ in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad(); out = model(inp); loss = criterion(out, tgt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_loss += loss.item()
            
        model.eval(); val_loss = 0; val_mse_real = 0
        with torch.no_grad():
            for inp, tgt, mv, _ in val_loader:
                inp, tgt, mv = inp.to(device), tgt.to(device), mv.to(device).unsqueeze(1)
                out = model(inp)
                val_loss += criterion(out, tgt).item()
                val_mse_real += torch.mean(((out*mv) - (tgt*mv))**2).item()
                
        train_loss /= len(train_loader); val_loss /= len(val_loader); val_mse_real /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Ep {epoch+1}: T_Loss={train_loss:.6f} | V_Loss={val_loss:.6f} | RealMSE={val_mse_real:.2e} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_loss:
            best_loss = val_loss; patience_cnt = 0
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss}, Path(config["run_dir"]) / 'best_model.pth')
            print("✅ Model Saved")
        else:
            patience_cnt += 1
            if patience_cnt >= config["early_stopping_patience"]: break

def run_evaluation(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis_dir = Path(config["run_dir"]) / "evaluation_visuals"; vis_dir.mkdir(exist_ok=True)
    
    all_clean, all_noise = gather_file_lists(config)
    random.seed(42); random.shuffle(all_clean)
    val_c = all_clean[int(len(all_clean) * (1 - config["real_data_val_split_ratio"])):]
    
    NOISE_GAIN = 20.0
    print(f"Evaluating with Fixed Noise Gain = {NOISE_GAIN}...")
    
    model = HybridTransformerCNN1D(config).to(device)
    model.load_state_dict(torch.load(Path(config["run_dir"]) / 'best_model.pth', map_location=device)['model_state_dict'])
    model.eval()
    
    results = []
    eval_dataset = RealAugmentedDataset(val_c, all_noise, is_train=False, fixed_noise_scale=NOISE_GAIN)
    loader = DataLoader(eval_dataset, batch_size=config["minibatch_size"], shuffle=False)
    
    total_mse = 0; count = 0
    
    with torch.no_grad():
        for inp, tgt, mv, fname in tqdm(loader):
            inp, tgt, mv = inp.to(device), tgt.to(device), mv.to(device).unsqueeze(1)
            out = model(inp)
            res_inp = (inp * mv).cpu().numpy()
            res_out = (out * mv).cpu().numpy()
            res_tgt = (tgt * mv).cpu().numpy()
            
            for i in range(len(fname)):
                mse = np.mean((res_out[i] - res_tgt[i])**2)
                total_mse += mse; count += 1
                if len(results) < 20:
                    results.append((res_inp[i], res_out[i], res_tgt[i], fname[i]))
    
    if count > 0: print(f"Avg Real MSE: {total_mse/count:.4e}")

    for i, (n, d, c, name) in enumerate(results):
        fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        ax[0].plot(n, color='gray', alpha=0.6, label='Input (Strong Noise)'); ax[0].plot(c, 'g--', alpha=0.5); ax[0].legend()
        ax[0].set_title(f"Input (Amp~{np.max(n):.1e})")
        ax[1].plot(d, 'b', label='Output'); ax[1].legend(); ax[1].set_title("Denoised")
        ax[2].plot(n-d, 'orange', label='Removed Noise'); ax[2].legend(); ax[2].set_title("Removed Component")
        ax[3].plot(d-c, 'r', label='Error'); ax[3].legend(); ax[3].set_title("Residual Error")
        plt.tight_layout(); plt.savefig(vis_dir / f"eval_{i}_{name}.png"); plt.close()
        np.save(vis_dir / f"denoised_{i}_{name}.npy", d)

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    run_training(config)
    run_evaluation(config)