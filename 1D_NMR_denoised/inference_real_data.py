import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from fid_denoising_unet_1d_2 import AttentionUNet1D

# ======================
# 1. 配置区域 (请根据实际情况修改)
# ======================

config = {
    # 训练好的模型路径 (.pth文件)
    # "model_path": "results_real_data_groupnorm_stable/best_model.pth",
    "model_path": "results_real_data_attention_ema/best_model.pth",
    
    
    # 真实数据所在的文件夹路径 (支持 .npy 文件，如果是txt需修改加载部分)
    "input_folder": r"E:\AI\PAN2025\DL_TDNMR_Inhance\Real_data_test",
    
    # 输出结果保存路径
    "output_folder": "results_inference_real_ema",
    
    # 训练时设定的核心窗口长度 (不要修改，必须与训练一致)
    "target_length": 4096,
    
    # 是否使用 CPU 或 GPU
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}



# ======================
# 3. 核心推理逻辑 (处理变长数据)
# ======================
def infer_signal(model, signal_np, device, target_len=4096):
    """
    对单个信号进行推理，处理长度不一致问题。
    """
    original_len = len(signal_np)
    
    # 1. 归一化 (记录最大值用于还原)
    max_val = np.max(np.abs(signal_np))
    if max_val < 1e-9: max_val = 1.0
    norm_sig = signal_np / max_val
    
    model.eval()
    denoised_result = np.zeros_like(norm_sig)
    
    with torch.no_grad():
        # 情况 A: 信号长度 小于或等于 训练窗口
        if original_len <= target_len:
            pad_len = target_len - original_len
            # 使用 reflect 填充，保持边缘特性
            padded_sig = np.pad(norm_sig, (0, pad_len), 'reflect')
            
            input_tensor = torch.from_numpy(padded_sig).float().unsqueeze(0).to(device) # [1, L]
            output_tensor = model(input_tensor) # [1, L]
            
            output_np = output_tensor.squeeze().cpu().numpy()
            denoised_result = output_np[:original_len] # 裁剪回原长度
            
        # 情况 B: 信号长度 大于 训练窗口 (使用滑窗法 + 重叠平均)
        else:
            stride = target_len // 2 # 50% 重叠
            counts = np.zeros_like(norm_sig) # 记录每个点被预测了多少次
            
            current_idx = 0
            while current_idx < original_len:
                # 确定窗口范围
                start = current_idx
                end = start + target_len
                
                # 如果到了最后一段，倒着取最后 target_len 长度
                if end > original_len:
                    start = original_len - target_len
                    end = original_len
                    
                chunk = norm_sig[start:end]
                
                input_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                output_tensor = model(input_tensor)
                output_chunk = output_tensor.squeeze().cpu().numpy()
                
                # 累加结果
                denoised_result[start:end] += output_chunk
                counts[start:end] += 1
                
                # 如果是最后一段，跳出循环
                if end == original_len:
                    break
                    
                current_idx += stride
            
            # 取平均
            denoised_result = denoised_result / np.maximum(counts, 1.0)

    # 2. 反归一化
    final_output = denoised_result * max_val
    return final_output

# ======================
# 4. 主程序
# ======================
def main():
    # 准备路径
    in_dir = Path(config["input_folder"])
    out_dir = Path(config["output_folder"])
    vis_dir = out_dir / "visualizations"
    npy_dir = out_dir / "denoised_data"
    
    vis_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Load model from: {config['model_path']}")
    print(f"Processing data from: {in_dir}")
    
    # 加载模型
    device = torch.device(config["device"])
    
    
    # model = UNet1D_GN().to(device)
    model = AttentionUNet1D().to(device)    
    
    
    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"Model file not found: {config['model_path']}")
        
    checkpoint = torch.load(config["model_path"], map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # 兼容只保存了state_dict的情况
    
    model.eval()
    print("Model loaded successfully.")
    
    # 获取文件列表 (假设是 .npy，如果是文本文件如 .txt 或 .csv，请修改此处)
    files = sorted(list(in_dir.glob("*.npy")))
    
    # 如果没有npy，尝试找txt
    if not files:
        files = sorted(list(in_dir.glob("*.txt")))
        print(f"Found {len(files)} .txt files.")
    else:
        print(f"Found {len(files)} .npy files.")

    if not files:
        print("No files found!")
        return

    # 循环推理
    for fpath in tqdm(files, desc="Inferring"):
        try:
            # --- 数据加载 ---
            if fpath.suffix == '.npy':
                raw_signal = np.load(fpath).astype(np.float32)
            elif fpath.suffix == '.txt':
                # 假设txt是单列数据
                raw_signal = np.loadtxt(fpath).astype(np.float32)
            else:
                continue
            
            # 确保是1D数组
            if raw_signal.ndim > 1:
                raw_signal = raw_signal.flatten()
                
            # --- 执行推理 ---
            denoised_signal = infer_signal(model, raw_signal, device, target_len=config["target_length"])
            
            # --- 保存结果 (.npy) ---
            save_name = fpath.stem + "_denoised.npy"
            np.save(npy_dir / save_name, denoised_signal)
            
            # --- 可视化绘图 ---
            # 只画前 2000 点或全长，避免太密集看不清
            plot_len = min(len(raw_signal), 4096 * 2) 
            
            plt.figure(figsize=(12, 6))
            
            # 子图1: 原始 vs 去噪 (重叠)
            plt.subplot(2, 1, 1)
            plt.plot(raw_signal[:plot_len], color='lightgray', label='Raw Real Data (Mixed)', linewidth=1)
            plt.plot(denoised_signal[:plot_len], color='blue', label='Denoised Output', linewidth=1.5, alpha=0.8)
            plt.title(f"File: {fpath.name} (Length: {len(raw_signal)})")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # 子图2: 滤除的成分 (近似噪声)
            plt.subplot(2, 1, 2)
            noise_removed = raw_signal - denoised_signal
            plt.plot(noise_removed[:plot_len], color='orange', label='Removed Component (Est. Noise)', linewidth=1)
            plt.title("Estimated Noise Component (Raw - Denoised)")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(vis_dir / (fpath.stem + "_vis.png"), dpi=150)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {fpath.name}: {e}")

    print(f"\nDone! Results saved to: {out_dir}")

if __name__ == "__main__":
    main()