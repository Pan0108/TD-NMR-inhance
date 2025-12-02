# DL_TDNMR_Inhance

基于深度学习的时间域核磁共振（TDNMR）信号增强项目，主要用于1D FID（自由感应衰减）信号的去噪增强处理。

## 项目概述

本项目专注于1D FID信号的去噪增强，通过先进的深度学习技术，提高核磁共振信号的质量和分辨率。项目使用1D Attention U-Net模型，结合注意力机制和残差学习，实现高效的信号去噪。信号截断重建主要作为训练数据预处理的辅助手段。

## 目录结构

```
DL_TDNMR_Inhance/
├── data/                  # 多种频率的信号数据
│   ├── noise100KHZ/       # 100kHz频率数据
│   ├── noise200KHz/       # 200kHz频率数据
│   └── noise1000khz/      # 1000kHz频率数据
├── 1D_NMR_denoised/       # 深度学习去噪模型
├── dataprocessing/        # 数据处理脚本（辅助训练）
├── Real_data_test/        # 真实数据测试集
└── results_*/             # 训练和推理结果
```

## 核心功能

### 1. 深度学习去噪增强
使用1D Attention U-Net模型对FID信号进行去噪增强，提高信号质量。

### 2. 批量数据处理
支持批量处理多个信号文件，提高处理效率。

### 3. 模型训练与推理
完整的模型训练和推理流程，支持模型保存和加载。

### 4. 可视化评估
生成原始信号与去噪后信号的对比图，便于评估模型效果。

### 5. 支持多种频率
处理不同频率的FID信号（100kHz, 200kHz, 500kHz, 1000kHz, 2083kHz等）。

## 模型架构

### 1D Attention U-Net

- **残差学习**：每个模块包含两个残差块，减少梯度消失问题
- **注意力机制**：在Skip Connection处加入Attention Gate，让模型关注重要特征
- **编码器-解码器结构**：4层下采样和上采样，逐步提取和恢复特征
- **EMA权重平滑**：使用指数移动平均，消除误差漂移

```
Input (1D Signal) → Encoder (4 levels) → Bottleneck → Decoder (4 levels) → Output (Denoised Signal)
```

### 损失函数

- **时间加权损失**：重点修复信号头部t=0的缺陷
- **频域损失**：使用L1损失约束频域特性
- **平滑损失**：使用总变分（TV）损失保持信号平滑

## 训练流程

1. **数据准备**：收集清洁和带噪声的FID信号
2. **信号截断重建**：对训练数据进行截断和重建预处理
3. **数据增强**：随机增益、噪声增强、位移等
4. **模型训练**：使用1D Attention U-Net进行训练
5. **模型评估**：在验证集上评估模型性能
6. **模型保存**：保存最佳模型权重

## 推理流程

1. **加载模型**：加载训练好的模型权重
2. **输入数据**：读取待处理的FID信号
3. **归一化**：对信号进行归一化处理
4. **变长信号处理**：使用滑窗法+重叠平均处理不同长度的信号
5. **模型推理**：使用1D Attention U-Net生成去噪信号
6. **反归一化**：恢复原始信号尺度
7. **结果保存**：保存去噪后的信号和可视化结果

## 主要脚本说明

### 1. 深度学习去噪模型

- **fid_denoisising_unet_1d.py**：1D Attention U-Net模型的核心实现
  - 模型定义、训练和评估
  - 支持多种频率数据处理
  - 包含EMA权重平滑和注意力机制

### 2. 数据处理脚本（辅助训练）

- **fid_reconstructed.py**：使用反向线性预测算法重建截断的FID信号
- **fid_cutdown.py**：对FID信号进行截断处理
- **systhitic_clean_FID.py**：系统地清洁FID信号
- **demo_np_read.py**：演示如何读取numpy格式的FID数据

## 使用方法

### 1. 模型训练

```python
# 修改fid_denoisising_unet_1d.py中的配置
config = {
    "experiment_name": "real_data_attention_ema",
    "run_dir": "results_real_data_attention_ema",
    "data_paths": {
        "base_processed": r"路径/到/处理后的数据",
        "bandwidths": ["100khz", "200khz", "500khz"]
    },
    "num_epochs": 1000,
    "minibatch_size": 64,
    "learning_rate": 5e-5,
    "use_ema": True
}

# 运行训练
python fid_denoisising_unet_1d.py
```

### 2. 模型推理

```python
# 修改推理脚本中的配置
config = {
    "model_path": "results_real_data_attention_ema/best_model.pth",
    "input_folder": r"Real_data_test",
    "output_folder": "results_inference_real_ema",
    "target_length": 4096,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}

# 运行推理
python inference_real_data.py
```

### 3. 数据预处理（训练辅助）

```python
# 修改fid_reconstructed.py中的参数
INPUT_ROOT_DIR = r'路径/到/原始数据'
OUTPUT_ROOT_DIR = r'路径/到/处理后的数据'
N_TRUNC = 30      # 截断点数

# 运行预处理
python fid_reconstructed.py
```

## 技术栈

- **深度学习框架**：PyTorch
- **核心模型**：1D Attention U-Net
- **数据处理**：NumPy, Pandas
- **可视化**：Matplotlib
- **性能优化**：EMA权重平滑
- **训练技巧**：数据增强、早期停止、学习率调整

## 结果评估

- **可视化对比**：原始信号 vs 去噪后信号 vs 移除的噪声
- **定量指标**：MSE（均方误差）
- **定性分析**：信号平滑度、峰值清晰度、噪声抑制效果

## 项目特点

- ✅ 专注于1D FID信号去噪增强
- ✅ 使用先进的1D Attention U-Net模型
- ✅ 结合注意力机制和残差学习
- ✅ 支持多种频率信号处理
- ✅ 完整的训练和推理流程
- ✅ 批量处理和可视化功能
- ✅ EMA权重平滑，提高模型稳定性

## 未来改进方向

- 优化模型结构，提高训练效率
- 支持更多类型的核磁共振信号
- 增加更多评估指标
- 提供GUI界面
- 支持实时处理

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请联系项目负责人。