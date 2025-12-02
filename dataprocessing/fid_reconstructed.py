# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# 1. 核心算法：反向线性预测 
#    这些函数保持不变，它们是整个流程的核心。
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

def _design_matrix_forward_lp(x, order):
    N = len(x)
    if order < 1 or N <= order: raise ValueError(f"order 必须 < len(x)")
    X = np.empty((N - order, order), dtype=np.complex128)
    for k in range(order): X[:, k] = x[order - 1 - k : N - 1 - k]
    return X, x[order:]

def fit_lp_ridge(x, order, ridge=1e-6):
    X, y = _design_matrix_forward_lp(x, order)
    XtX = X.conj().T @ X
    Xty = X.conj().T @ (-y)
    lam = ridge * np.trace(XtX).real / max(order, 1)
    A = XtX + lam * np.eye(order, dtype=np.complex128)
    a = np.linalg.solve(A, Xty)
    mse = float(np.mean(np.abs(y + X @ a) ** 2))
    return a, mse

def select_order_aic(x, min_order=8, max_order=64, step=2, ridge=1e-6):
    N = len(x)
    max_order = min(max_order, max(1, N - 2))
    candidates = [o for o in range(min_order, max_order + 1, step) if o < N - 1]
    if not candidates: candidates = [min(8, max(1, N // 4))]
    best, best_aic, best_mse = None, np.inf, None
    for o in candidates:
        try:
            a, mse = fit_lp_ridge(x, o, ridge=ridge)
            sigma2 = max(mse, 1e-18)
            aic = 2 * o + (N - o) * np.log(sigma2)
            if aic < best_aic: best_aic, best, best_mse = aic, a, mse
        except Exception: continue
    if best is None:
        best, best_mse = fit_lp_ridge(x, order=min(4, N - 2), ridge=ridge)
    return best, best_mse, len(best)

def forward_predict(x, a, n_future):
    order = len(a)
    buf = np.array([x[-1 - k] for k in range(order)], dtype=np.complex128)
    preds = np.empty(n_future, dtype=np.complex128)
    for i in range(n_future):
        yhat = -np.dot(a, buf)
        preds[i], buf = yhat, np.roll(buf, 1); buf[0] = yhat
    return preds

def back_predict_missing_prefix(x, n_missing, order, auto_order, ridge):
    r = x[::-1].astype(np.complex128)
    if auto_order or order is None:
        a, mse, chosen_order = select_order_aic(r, ridge=ridge)
    else:
        chosen_order = min(order, len(r) - 2) if len(r) > 2 else 1
        a, mse = fit_lp_ridge(r, chosen_order, ridge=ridge)
    preds_r = forward_predict(r, a, n_missing)
    return preds_r[::-1], {"order": chosen_order, "mse": mse}

def truncate_and_reconstruct(fid, n_trunc, order, auto_order, ridge):
    if n_trunc <= 0: return fid.copy(), {"order": None, "mse": None}
    if len(fid) <= n_trunc + 8:
        print(f"警告: 数据点过少 ({len(fid)})，无法截断 {n_trunc} 点并训练。跳过重建。")
        return fid.copy(), {"order": -1, "mse": -1}
    known = fid[n_trunc:]
    known_target = fid[n_trunc-1:]   #拿到用于校准的真实目标值，即截断前的最后一个点
    pred, info = back_predict_missing_prefix(known, n_missing=n_trunc, order=order, auto_order=auto_order, ridge=ridge)
    if np.abs(pred[-1]) > 0:
        scale = known_target[0] / pred[-1]   # 使用新的、更正确的逻辑来计算缩放因子
        pred *= scale
    return np.concatenate([pred, known]), info

# =============================================================================
# 2. 批量处理函数 (框架来自您参考的代码)
# =============================================================================
def batch_process_fid_reconstruction(input_root, output_root, n_trunc, order, auto_order, ridge):
    """
    批量处理文件夹中的所有CSV文件。
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    print(f"开始处理数据，输入根目录: {input_root}")
    print(f"处理后数据将保存至: {output_root}")
    print("-" * 30)
    
    csv_files = list(input_root.glob('**/*.csv'))
    if not csv_files:
        print("错误: 在输入目录中没有找到任何.csv文件。请检查路径和文件结构。")
        return None, None, None

    last_original_fid = None
    last_reconstructed_fid = None
    last_filename = ""

    for csv_path in tqdm(csv_files, desc="总进度"):
        try:
            # 1. 加载原始复数FID
            original_fid = load_fid_csv(csv_path)
            
            # 2. 执行核心重建算法
            reconstructed_fid, info = truncate_and_reconstruct(
                original_fid,
                n_trunc=n_trunc,
                order=order,
                auto_order=auto_order,
                ridge=ridge
            )
            
            # 3. 计算重建后信号的模值
            reconstructed_magnitude = np.abs(reconstructed_fid)
            
            # 4. 准备输出路径和新文件名
            relative_path = csv_path.relative_to(input_root)
            # 使用父文件夹名作为前缀，如果文件就在根目录则无前缀
            prefix = relative_path.parts[0] + '_' if len(relative_path.parts) > 1 else ''
            new_filename = f"{prefix}{csv_path.stem}_reconstructed_magnitude.npy"
            
            output_dir = output_root / relative_path.parent
            output_filepath = output_dir / new_filename
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. 保存模值数据为.npy文件
            np.save(output_filepath, reconstructed_magnitude)

            # 保存最后一个文件的信息用于最终可视化
            last_original_fid = original_fid
            last_reconstructed_fid = reconstructed_fid
            last_filename = csv_path.name

        except Exception as e:
            print(f"处理文件 {csv_path} 时发生严重错误: {e}")

    print("-" * 30)
    print("所有文件处理完成！")
    return last_original_fid, last_reconstructed_fid, last_filename

# =============================================================================
# 3. 可视化函数
# =============================================================================
def plot_compare(original, reconstructed, n_trunc, title="FID Reconstruction Comparison"):
    """绘制原始信号和重建信号的实部、虚部和模值对比图。"""
    t = np.arange(len(original))
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=16)

    # 实部
    axs[0].plot(t, original.real, label="原始实部", alpha=0.7)
    axs[0].plot(t, reconstructed.real, "--", label="重建实部", alpha=0.9)
    axs[0].axvline(n_trunc - 0.5, color='k', ls=':', alpha=0.6, label=f'截断点 n={n_trunc}')
    axs[0].legend(); axs[0].set_ylabel("实部")
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 虚部
    axs[1].plot(t, original.imag, label="原始虚部", alpha=0.7)
    axs[1].plot(t, reconstructed.imag, "--", label="重建虚部", alpha=0.9)
    axs[1].axvline(n_trunc - 0.5, color='k', ls=':', alpha=0.6)
    axs[1].legend(); axs[1].set_ylabel("虚部")
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    # 模值
    axs[2].plot(t, np.abs(original), label="原始模值", alpha=0.7)
    axs[2].plot(t, np.abs(reconstructed), "--", label="重建模值", alpha=0.9)
    axs[2].axvline(n_trunc - 0.5, color='k', ls=':', alpha=0.6)
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
    #    Windows 示例: r"C:\data\nmr\raw_data"
    #    macOS/Linux 示例: "/home/user/data/raw_data"
    INPUT_ROOT_DIR = r'E:\AI\PAN2025\DL_TDNMR_Inhance\data\noise2083KHZ\clean'
    OUTPUT_ROOT_DIR = r'E:\AI\PAN2025\DL_TDNMR_Inhance\data\processed_data_relaxation\2083khz\clean'
    
    # 2. 设置重建参数
    N_TRUNC = 30      # 要截断并重建的点数 ,100khz-->clean=25;noisy=10   200khz-->clean=25, noisy=15  500khz-->clean=25,noisy=20;1000khz--> clean=23, noisy=20; 2000khz--> clean=30; noisy=25
                        #333.33khz ---> clean= 25, noisy = 20
    # 3. 设置模型阶数
    #    - 设为 None 来启用 AIC 自动选阶 (推荐)
    #    - 或手动指定一个整数，例如 40
    ORDER = None
    
    # 4. 其他高级参数
    RIDGE = 1e-6     # 岭回归正则化强度，通常无需修改
    
    # 5. 是否在全部处理完成后，显示最后一个文件的可视化对比图
    VISUALIZE_LAST_FILE = True
    
    # --- 配置结束 ---

    # 执行批量处理
    original, reconstructed, filename = batch_process_fid_reconstruction(
        input_root=INPUT_ROOT_DIR,
        output_root=OUTPUT_ROOT_DIR,
        n_trunc=N_TRUNC,
        order=ORDER,
        auto_order=(ORDER is None), # 如果手动指定了ORDER，则自动禁用auto_order
        ridge=RIDGE
    )

    # 如果启用了可视化，并且成功处理了至少一个文件，则绘图
    if VISUALIZE_LAST_FILE and original is not None:
        print(f"\n正在可视化最后一个成功处理的文件: {filename}")
        plot_compare(original, reconstructed, N_TRUNC, title=f"文件 '{filename}' 的重建结果")
    elif VISUALIZE_LAST_FILE:
        print("\n没有成功处理任何文件，无法进行可视化。")