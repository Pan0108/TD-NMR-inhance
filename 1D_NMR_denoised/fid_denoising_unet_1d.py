# relax_denoising_unet_ultimate_v18.py
# 描述: 终极优化版本。
#       1. 架构升级: Attention U-Net (在 Skip Connection 处加入 Attention Gate)。
#       2. 训练技巧: 引入 EMA (Exponential Moving Average) 权重平滑，消除误差漂移。
#       3. 损失函数: 引入 Time-Weighted Loss，强迫模型修复头部 t=0 的缺陷。

import os
import time
import random
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ======================
# CONFIGURATION SETTINGS
# ======================
config = {
    "experiment_name": "real_data_attention_ema",
    "run_dir": "results_real_data_attention_ema",
    
    "data_paths": {
        "base_processed": r"E:\AI\PAN2025\DL_TDNMR_Inhance\data\processed_data_relaxation",
        "bandwidths": ["100khz", "200khz", "333khz", "500khz", "1000khz", "2083khz"],
    },

    "num_acquisitions": 128,
    "target_length": 4096,
    "real_data_val_split_ratio": 0.2, 
    
    "resume_training": False,
    "checkpoint_path": "results_real_data_attention_ema/best_model.pth",
    "num_epochs": 1000, 
    "minibatch_size": 64, 
    "early_stopping_patience": 50, 
    
    "learning_rate": 5e-5, 
    "optimizer_weight_decay": 1e-5,
    "base_channels": 32,
    
    # 训练时的随机范围
    "aug_gain_range": [0.5, 2.0], 
    "aug_noise_gain_range": [0.5, 10.0],
    "aug_shift_range": 5, 
    
    # EMA 参数
    "use_ema": True,
    "ema_decay": 0.999, # 衰减率，越高越平滑
}

# ======================
# HELPER: EMA (Exponential Moving Average)
# ======================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        # 注册模型参数
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
                self.original[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]
        self.original = {}

# ======================
# HELPER: Signal Processing
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
        return np.pad(signal, (0, pad_len), 'reflect') 
    else:
        return signal

# ======================
# LOSS FUNCTIONS (Weighted + Combined)
# ======================
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        diff = torch.abs(x[:, 1:] - x[:, :-1])
        return torch.mean(diff)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.1, head_weight=5.0):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.head_weight = head_weight 
        
        self.l1 = nn.L1Loss()
        self.tv = TVLoss()

    def forward(self, inputs, targets):
        # 【修复】自动处理维度，确保是 (Batch, Channel, Length)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        # 现在可以安全解包了
        B, C, L = inputs.shape
        device = inputs.device
        
        # 1. 时域损失 (Weighted MSE)
        # 线性衰减权重: [5.0 -> 1.0] for first 200 points
        w = torch.ones(L, device=device)
        head_len = 200
        # 防止信号长度小于200的情况报错
        actual_head_len = min(L, head_len)
        w[:actual_head_len] = torch.linspace(self.head_weight, 1.0, actual_head_len, device=device)
        
        diff = (inputs - targets) ** 2
        # 扩展权重维度以匹配 (B, C, L)
        time_loss = torch.mean(diff * w.view(1, 1, -1))
        
        # 2. 频域损失 (L1)
        input_fft = torch.fft.rfft(inputs, dim=-1, norm='ortho')
        target_fft = torch.fft.rfft(targets, dim=-1, norm='ortho')
        input_mag = torch.abs(input_fft)
        target_mag = torch.abs(target_fft)
        freq_loss = self.l1(input_mag, target_mag)
        
        # 3. 平滑损失 (TV)
        tv_loss = self.tv(inputs)
        
        return time_loss + (self.alpha * freq_loss) + (self.beta * tv_loss)

# ======================
# MODEL: Attention U-Net
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: residual = self.downsample(x)
        out += residual
        return self.relu(out)

class AttentionBlock(nn.Module):
    """
    Attention Gate: 
    让 Decoder (g) 决定 Encoder 特征 (x) 中哪些部分是重要的。
    有效抑制 Skip Connection 传过来的背景噪声。
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # g: gating signal (from decoder)
        # x: skip connection signal (from encoder)
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        base_channels = config["base_channels"]
        
        # Encoder
        self.enc1 = self._block(in_channels, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = self._block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = self._block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._block(base_channels * 8, base_channels * 16)
        
        # Decoder with Attention Gates
        # Layer 4
        self.upconv4 = nn.ConvTranspose1d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=base_channels * 8, F_l=base_channels * 8, F_int=base_channels * 4)
        self.dec4 = self._block(base_channels * 16, base_channels * 8)
        
        # Layer 3
        self.upconv3 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=base_channels * 4, F_l=base_channels * 4, F_int=base_channels * 2)
        self.dec3 = self._block(base_channels * 8, base_channels * 4)
        
        # Layer 2
        self.upconv2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=base_channels * 2, F_l=base_channels * 2, F_int=base_channels)
        self.dec2 = self._block(base_channels * 4, base_channels * 2)
        
        # Layer 1
        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=base_channels, F_l=base_channels, F_int=base_channels // 2)
        self.dec1 = self._block(base_channels * 2, base_channels)
        
        self.final = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(ResidualBlock1D(in_ch, out_ch), ResidualBlock1D(out_ch, out_ch))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with Attention
        d4_up = self.upconv4(b)
        e4_att = self.att4(g=d4_up, x=e4) # Attention Filtering
        d4 = self.dec4(torch.cat([d4_up, e4_att], dim=1))
        
        d3_up = self.upconv3(d4)
        e3_att = self.att3(g=d3_up, x=e3)
        d3 = self.dec3(torch.cat([d3_up, e3_att], dim=1))
        
        d2_up = self.upconv2(d3)
        e2_att = self.att2(g=d2_up, x=e2)
        d2 = self.dec2(torch.cat([d2_up, e2_att], dim=1))
        
        d1_up = self.upconv1(d2)
        e1_att = self.att1(g=d1_up, x=e1)
        d1 = self.dec1(torch.cat([d1_up, e1_att], dim=1))
        
        return self.final(d1).squeeze(1)

# ======================
# DATASET
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
                
                shift = random.randint(-config["aug_shift_range"], config["aug_shift_range"])
                if shift > 0:
                    base_clean = np.roll(base_clean, shift)
                    base_clean[:shift] = base_clean[shift] 
                elif shift < 0:
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
# TRAINING ROUTINE
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
    
    # 使用 Attention U-Net
    model = AttentionUNet1D().to(device)
    
    # 初始化 EMA
    if config["use_ema"]:
        ema = EMA(model, config["ema_decay"])
        print(f"EMA Enabled with decay {config['ema_decay']}")
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["optimizer_weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 使用加权 Loss
    criterion = CombinedLoss(alpha=0.2, beta=0.1, head_weight=5.0).to(device)
    
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
            optimizer.step()
            
            if config["use_ema"]: ema.update(model) # 更新 EMA
            
            train_loss += loss.item()
            
        # 验证时使用 EMA 权重
        if config["use_ema"]: ema.apply_shadow(model)
        
        model.eval(); val_loss = 0; val_mse_real = 0
        with torch.no_grad():
            for inp, tgt, mv, _ in val_loader:
                inp, tgt, mv = inp.to(device), tgt.to(device), mv.to(device).unsqueeze(1)
                out = model(inp)
                val_loss += criterion(out, tgt).item()
                val_mse_real += torch.mean(((out*mv) - (tgt*mv))**2).item()
        
        # 恢复原始权重继续训练
        if config["use_ema"]: ema.restore(model)
                
        train_loss /= len(train_loader); val_loss /= len(val_loader); val_mse_real /= len(val_loader)
        scheduler.step(val_loss)
        print(f"Ep {epoch+1}: T_Loss={train_loss:.6f} | V_Loss={val_loss:.6f} | RealMSE={val_mse_real:.2e} | LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_loss:
            best_loss = val_loss; patience_cnt = 0
            # 保存模型时，如果有 EMA，建议保存 EMA 的权重
            save_state = model.state_dict()
            if config["use_ema"]:
                ema.apply_shadow(model)
                save_state = model.state_dict()
                ema.restore(model)
                
            torch.save({'model_state_dict': save_state, 'val_loss': val_loss}, Path(config["run_dir"]) / 'best_model.pth')
            print("✅ Model Saved (EMA)")
        else:
            patience_cnt += 1
            if patience_cnt >= config["early_stopping_patience"]: break

def run_evaluation(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vis_dir = Path(config["run_dir"]) / "evaluation_visuals"; vis_dir.mkdir(exist_ok=True)
    
    all_clean, all_noise = gather_file_lists(config)
    random.seed(42); random.shuffle(all_clean)
    val_c = all_clean[int(len(all_clean) * (1 - config["real_data_val_split_ratio"])):]
    
    NOISE_GAIN = 10.0
    print(f"Evaluating with Fixed Noise Gain = {NOISE_GAIN}...")
    
    model = AttentionUNet1D().to(device)
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