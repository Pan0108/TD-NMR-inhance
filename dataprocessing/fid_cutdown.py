# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# 1. 数据加载与核心处理算法
# =============================================================================

def load_fid_csv(path):
    """健壮地读取 CSV 文件为复数 FID 信号。"""
    try:
        import pandas as pd
        df = pd.read_csv(path, comment="#")
        cols_lower = [str(c).lower() for c in df.columns]
        real_idx, imag_idx = None, None
        for i, c in enumerate(cols_lower):
            if 'real' in c and real_idx is None: real_idx = i
            if ('imag' in c or 'imaginary' in c) and imag_idx is None: imag_idx = i
        if real_idx is None or imag_idx is None:
            real = df.iloc[:, 0].astype(np.float64).to_numpy()
            imag = df.iloc[:, 1].astype(np.float64).to_numpy()
        else:
            real = df.iloc[:, real_idx].astype(np.float64).to_numpy()
            imag = df.iloc[:, imag_idx].astype(np.float64).to_numpy()
    except Exception:
        arr = np.loadtxt(path, delimiter=",")
        real, imag = arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    return real + 1j * imag

def truncate_signal(fid, n_trunc):
    """
    仅截断信号，不进行重建。
    :param fid: 原始复数信号
    :param n_trunc: 需要截断的前 n 个点
    :return: 截断后的复数信号 (长度为 len(fid) - n_trunc)
    """
    if n_trunc <= 0:
        return fid.copy()
    
    if len(fid) <= n_trunc:
        print(f"警告: 数据点过少 ({len(fid)})，无法截断 {n_trunc} 点。返回空数组。")
        return np.array([], dtype=np.complex128)
    
    # 直接切片，去掉前 n_trunc 个点
    return fid[n_trunc:]

# =============================================================================
# 2. 批量处理函数
# =============================================================================
def batch_process_fid_truncation(input_root, output_root, n_trunc):
    """
    批量处理文件夹中的所有CSV文件：截断 -> 取模值 -> 保存。
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    print(f"开始处理数据 (仅截断模式)，输入根目录: {input_root}")
    print(f"处理后数据将保存至: {output_root}")
    print("-" * 30)
    
    csv_files = list(input_root.glob('**/*.csv'))
    if not csv_files:
        print("错误: 在输入目录中没有找到任何.csv文件。请检查路径和文件结构。")
        return None, None, None

    last_original_fid = None
    last_truncated_fid = None
    last_filename = ""

    for csv_path in tqdm(csv_files, desc="总进度"):
        try:
            # 1. 加载原始复数FID
            original_fid = load_fid_csv(csv_path)
            
            # 2. 执行截断 (不重建)
            truncated_fid = truncate_signal(original_fid, n_trunc=n_trunc)
            
            if len(truncated_fid) == 0:
                continue

            # 3. 计算截断后信号的模值
            truncated_magnitude = np.abs(truncated_fid)
            
            # 4. 准备输出路径和新文件名
            relative_path = csv_path.relative_to(input_root)
            # 使用父文件夹名作为前缀
            prefix = relative_path.parts[0] + '_' if len(relative_path.parts) > 1 else ''
            # 文件名后缀改为 truncated_magnitude
            new_filename = f"{prefix}{csv_path.stem}_truncated_magnitude.npy"
            
            output_dir = output_root / relative_path.parent
            output_filepath = output_dir / new_filename
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. 保存模值数据为.npy文件
            np.save(output_filepath, truncated_magnitude)

            # 保存最后一个文件的信息用于最终可视化
            last_original_fid = original_fid
            last_truncated_fid = truncated_fid
            last_filename = csv_path.name

        except Exception as e:
            print(f"处理文件 {csv_path} 时发生严重错误: {e}")

    print("-" * 30)
    print("所有文件处理完成！")
    return last_original_fid, last_truncated_fid, last_filename

# =============================================================================
# 3. 可视化函数
# =============================================================================
def plot_compare(original, truncated, n_trunc, title="FID Truncation Comparison"):
    """
    绘制原始信号和截断后信号的对比图。
    截断后的信号将在时间轴上右移，以对齐原始数据的尾部。
    """
    t_orig = np.arange(len(original))
    # 截断后的信号对应的时间轴起点是 n_trunc
    t_trunc = np.arange(n_trunc, len(original)) 
    
    # 确保长度匹配（防止意外）
    if len(t_trunc) != len(truncated):
        min_len = min(len(t_trunc), len(truncated))
        t_trunc = t_trunc[:min_len]
        truncated = truncated[:min_len]

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=16)

    # 实部
    axs[0].plot(t_orig, original.real, label="原始实部", alpha=0.5, color='gray')
    axs[0].plot(t_trunc, truncated.real, label="截断后实部", alpha=0.9, color='blue')
    axs[0].axvline(n_trunc - 0.5, color='red', ls=':', alpha=0.8, label=f'截断点 n={n_trunc}')
    axs[0].legend(); axs[0].set_ylabel("实部")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 虚部
    axs[1].plot(t_orig, original.imag, label="原始虚部", alpha=0.5, color='gray')
    axs[1].plot(t_trunc, truncated.imag, label="截断后虚部", alpha=0.9, color='orange')
    axs[1].axvline(n_trunc - 0.5, color='red', ls=':', alpha=0.8)
    axs[1].legend(); axs[1].set_ylabel("虚部")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # 模值
    axs[2].plot(t_orig, np.abs(original), label="原始模值", alpha=0.5, color='gray')
    axs[2].plot(t_trunc, np.abs(truncated), label="截断后模值", alpha=0.9, color='green')
    axs[2].axvline(n_trunc - 0.5, color='red', ls=':', alpha=0.8)
    axs[2].legend(); axs[2].set_ylabel("模值"); axs[2].set_xlabel("数据点索引")
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# =============================================================================
# 4. 主程序入口和参数配置
# =============================================================================
if __name__ == '__main__':
    
    # --- 请在这里配置您的参数 ---
    
    # 1. 设置输入和输出文件夹路径
    INPUT_ROOT_DIR = r'E:\AI\PAN2025\DL_TDNMR_Inhance\Real_data_test\100khz'
    OUTPUT_ROOT_DIR = r'E:\AI\PAN2025\DL_TDNMR_Inhance\Real_data_test' # 修改了输出目录名以避免混淆
    
    # 2. 设置截断参数
    N_TRUNC = 25       # 要截断并重建的点数 ,100khz-->clean=25;noisy=10   200khz-->clean=25, noisy=15  500khz-->clean=25,noisy=20;1000khz--> clean=23, noisy=20; 2000khz--> clean=30; noisy=25
                        #333.33khz ---> clean= 25, noisy = 20
    
    # 3. 是否在全部处理完成后，显示最后一个文件的可视化对比图
    VISUALIZE_LAST_FILE = True
    
    # --- 配置结束 ---

    # 执行批量处理 (不再需要 order, ridge 等参数)
    original, truncated, filename = batch_process_fid_truncation(
        input_root=INPUT_ROOT_DIR,
        output_root=OUTPUT_ROOT_DIR,
        n_trunc=N_TRUNC
    )

    # 如果启用了可视化，并且成功处理了至少一个文件，则绘图
    if VISUALIZE_LAST_FILE and original is not None:
        print(f"\n正在可视化最后一个成功处理的文件: {filename}")
        plot_compare(original, truncated, N_TRUNC, title=f"文件 '{filename}' 的截断结果 (移除前 {N_TRUNC} 点)")
    elif VISUALIZE_LAST_FILE:
        print("\n没有成功处理任何文件，无法进行可视化。")