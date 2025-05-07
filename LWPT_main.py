# -*- coding: utf-8 -*-
"""
此脚本用于训练 LWPT 模型进行时间序列降噪，执行两组消融实验：
1. Focus 机制: Focus vs. NoFocus
2. 损失函数: SmoothL1 Loss vs. L1 Loss
输出直接保存在数据集同目录下的 LWPTXRSY 文件夹中，通过文件名区分。
改动较小版。
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from torch.utils.data import Dataset, DataLoader
import traceback

# 尝试添加自定义模块的路径 - !! 请确保路径正确 !!
# 尝试添加自定义模块的路径 (使用相对路径)
try:
    # 直接添加相对于当前工作目录的 'Code' 文件夹路径
    # **重要假设**: 你运行 `python LWPT_main.py` 命令时，
    #            你正位于 `Learnable-Wavelet-Tran...` 这个目录下。
    relative_code_path = 'Code' # <--- 直接写 'Code' (大写 C)
    sys.path.append(relative_code_path)
    print(f"已添加相对路径到 sys.path: '{relative_code_path}'")

    # 现在尝试导入模块
    import NeuralDWAV
    import Util_NeuralDWAV as Utils
    print("成功导入自定义模块 NeuralDWAV 和 Util_NeuralDWAV。")

except ImportError as e:
    print(f"导入自定义模块时出错: {e}")
    # 更新错误提示，强调当前工作目录和文件夹大小写
    print(f"请确保你运行脚本时，当前工作目录是 'Learnable-Wavelet-Tran...' (即包含 LWPT_main.py 的目录)。")
    print(f"并确保该目录下存在名为 'Code' (大写C) 的子文件夹，且其中包含所需的 .py 文件。")
    # sys.exit(1) # 如果需要，取消注释以退出

# 设置计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"运行设备: {device}")

def find_nearest_power_of_two(n):
    """找到大于等于n的最小的2的幂次方。"""
    power = 1
    while power < n:
        power *= 2
    return power

# --- TimeSeriesDataset 类 (与你原版一致，包含归一化) ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, Np, original_length=None):
        self.data = data.copy(deep=True)
        self.Np = Np
        self.total_samples = len(self.data)
        self.original_length = original_length if original_length is not None else self.total_samples
        self.scaling_factors = {}
        self.column_name = self.data.columns[0]
        self._normalize_data()

    def _normalize_data(self):
        self.original_data_unnormalized = self.data.copy(deep=True)
        col = self.column_name
        original_values = self.original_data_unnormalized[col].iloc[:self.original_length]
        mean = original_values.mean()
        std = original_values.std()
        if std == 0:
            std = 1e-6
        self.scaling_factors[col] = {'mean': mean, 'std': std}
        self.data.loc[:, col] = (self.data[col] - mean) / std

    def denormalize(self, normalized_data, column_name=None):
        col = column_name if column_name is not None else self.column_name
        if col not in self.scaling_factors:
            raise ValueError(f"未找到列 '{col}' 的缩放因子。")
        factors = self.scaling_factors[col]
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.cpu().numpy()
        return normalized_data * factors['std'] + factors['mean']

    def __len__(self):
        return (self.total_samples + self.Np - 1) // self.Np

    def __getitem__(self, idx):
        start = idx * self.Np
        end = min(start + self.Np, self.total_samples)
        chunk = self.data.iloc[start:end].copy()
        actual_len = len(chunk)

        if actual_len < self.Np:
            pad_length = self.Np - actual_len
            if actual_len > 0:
                 mirror_chunk = chunk.iloc[-(pad_length % actual_len):].iloc[::-1].copy()
                 while len(mirror_chunk) < pad_length:
                     if len(mirror_chunk) == 0: break
                     mirror_chunk = pd.concat([mirror_chunk, mirror_chunk.iloc[::-1]], axis=0)
                 mirror_chunk = mirror_chunk.iloc[:pad_length]
            else:
                 mirror_chunk = pd.DataFrame(np.zeros((pad_length, chunk.shape[1])), columns=chunk.columns)
            chunk = pd.concat([chunk, mirror_chunk], axis=0)

        if len(chunk) != self.Np:
             padding_needed = self.Np - len(chunk)
             if padding_needed > 0:
                  zero_padding = pd.DataFrame(np.zeros((padding_needed, chunk.shape[1])), columns=chunk.columns, index=range(len(chunk), self.Np))
                  chunk = pd.concat([chunk, zero_padding], axis=0)
             elif padding_needed < 0:
                  chunk = chunk.iloc[:self.Np]

        return torch.tensor(chunk.values, dtype=torch.double).view(1, -1)

# --- DataLoader 获取函数 (与你原版一致) ---
def get_dataloader(dataset, batch_size, num_workers=0, pin_memory=False):
     # 根据你的原版，可能 num_workers=4， 但 0 更安全，尤其在 Windows
     # 如果需要，可以将 num_workers 改回 4
    effective_num_workers = 0 # 使用 0 避免多进程问题
    if torch.cuda.is_available() and pin_memory:
        effective_pin_memory = True
    else:
        effective_pin_memory = False

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=effective_num_workers, pin_memory=effective_pin_memory
    )


# --- 模型和优化器初始化函数 (与你原版一致) ---
def initialize_model(Np, input_level=6, input_archi="WPT"):
    # 检查 NeuralDWAV 是否已成功导入
    if 'NeuralDWAV' not in globals():
        raise ImportError("错误: NeuralDWAV 模块未能导入，无法初始化模型。")
    model = NeuralDWAV.NeuralDWAV(Input_Size=Np, Input_Level=input_level, Input_Archi=input_archi).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    return model, optimizer

# --- 损失计算函数 (根据 mode 计算范围，与上版类似) ---
def compute_loss(model, x_batch, loss_fn, Lambda, original_length_in_split, batch_idx, Np, mode='Focus'):
    # 注意: 此函数依赖于 model.T, model.iT, model.L1_sum 方法
    Emb = model.T(x_batch)
    Emb_copy = [e.clone() for e in Emb]
    reconstructed = model.iT(Emb_copy)

    start_sample_idx = batch_idx * Np
    end_sample_idx = start_sample_idx + Np
    valid_original_start = max(0, start_sample_idx)
    valid_original_end = min(original_length_in_split, end_sample_idx)
    valid_points_in_chunk = max(0, valid_original_end - valid_original_start)
    start_offset_in_batch = max(0, valid_original_start - start_sample_idx)
    end_offset_in_batch = start_offset_in_batch + valid_points_in_chunk

    reconstruction_loss = torch.tensor(0.0, device=device, dtype=torch.double)
    if mode == 'Focus':
        if valid_points_in_chunk > 0:
            original_part_x = x_batch[:, :, start_offset_in_batch:end_offset_in_batch]
            original_part_reconstructed = reconstructed[:, :, start_offset_in_batch:end_offset_in_batch]
            reconstruction_loss = loss_fn(original_part_reconstructed, original_part_x)
    elif mode == 'NoFocus':
        reconstruction_loss = loss_fn(reconstructed, x_batch)
    else:
        raise ValueError(f"未知的模式: {mode}")

    regularization_loss = Lambda * model.L1_sum(Emb)
    total_loss = reconstruction_loss + regularization_loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"警告: 损失计算结果无效 (NaN/Inf)。")
        return torch.tensor(float('inf'), device=device, dtype=torch.double)
    return total_loss

# --- 单列处理核心函数 (添加 mode 和 loss_type 参数, 修改输出路径) ---
def process_single_column(col, full_data_col, original_length, time_column, Np, epochs, patience, batch_size,
                          # 修改: 传入基础输出目录，而不是让函数创建
                          output_dir,
                          Lambda,
                          # 新增参数:
                          mode='Focus',
                          loss_type='SmoothL1'):
    """
    处理单列时间序列数据，根据 mode 和 loss_type 配置。
    结果直接保存到 output_dir，文件名包含配置信息。
    """
    run_identifier = f"列={col}, 模式={mode}, 损失={loss_type}"
    print(f"--- 开始处理: {run_identifier} ---")
    column_start_time = time.time()

    # --- 数据准备与划分 (与原版逻辑相同) ---
    split_idx = int(0.7 * original_length)
    train_data = full_data_col.iloc[:split_idx].copy() if split_idx > 0 else pd.DataFrame(columns=full_data_col.columns)
    val_data = full_data_col.iloc[split_idx:original_length].copy()
    full_dataset_data = full_data_col.copy()
    train_len = len(train_data)
    val_len = len(val_data)

    train_dataset = TimeSeriesDataset(train_data, Np, original_length=train_len) if train_len > 0 else None
    val_dataset = TimeSeriesDataset(val_data, Np, original_length=val_len) if val_len > 0 else None
    full_dataset = TimeSeriesDataset(full_dataset_data, Np, original_length=original_length)

    if train_dataset is None or len(train_dataset) == 0:
         print(f"警告: {run_identifier} 训练集为空，跳过。")
         return
    # 允许没有验证集的情况

    train_loader = get_dataloader(train_dataset, batch_size)
    val_loader = get_dataloader(val_dataset, batch_size) if val_dataset and len(val_dataset) > 0 else None
    full_loader = get_dataloader(full_dataset, batch_size)

    # --- 模型初始化 (与原版逻辑相同) ---
    LWPT, optimizer = initialize_model(Np)

    # --- 选择损失函数 (小改动) ---
    if loss_type == 'SmoothL1':
        loss_fn = nn.SmoothL1Loss()
    elif loss_type == 'L1':
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")
    # print(f"使用 {loss_type} 损失函数。") # 减少输出

    # --- 学习率调度器 (与原版逻辑相同) ---
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        print(f"警告: {run_identifier} 训练加载器为空，跳过。")
        return
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=steps_per_epoch
    )

    # --- 训练循环 (与原版逻辑类似) ---
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # print(f"开始训练 {run_identifier}...") # 减少输出
    for epoch in range(epochs):
        LWPT.train()
        epoch_train_loss = 0
        train_batch_count = 0
        for batch_idx, x_batch_train in enumerate(train_loader):
            x_batch_train = x_batch_train.to(device)
            optimizer.zero_grad()
            loss = compute_loss(LWPT, x_batch_train, loss_fn, Lambda, train_len, batch_idx, Np, mode)

            if torch.isfinite(loss) and (loss.item() > 0 or mode == 'NoFocus'):
                 loss.backward()
                 optimizer.step()
                 scheduler.step()
                 epoch_train_loss += loss.item()
                 train_batch_count += 1
            # 忽略无效损失或Focus模式下的0损失

        average_train_loss = epoch_train_loss / train_batch_count if train_batch_count > 0 else 0
        train_losses.append(average_train_loss)

        # --- 验证阶段 ---
        average_val_loss = float('inf')
        if val_loader:
            LWPT.eval()
            epoch_val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for batch_idx, x_batch_val in enumerate(val_loader):
                    x_batch_val = x_batch_val.to(device)
                    val_loss_item = compute_loss(LWPT, x_batch_val, loss_fn, Lambda, val_len, batch_idx, Np, mode)
                    if torch.isfinite(val_loss_item) and (val_loss_item.item() > 0 or mode == 'NoFocus'):
                        epoch_val_loss += val_loss_item.item()
                        val_batch_count += 1

            average_val_loss = epoch_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            val_losses.append(average_val_loss)

            # print(f"Epoch {epoch+1}/{epochs} | {run_identifier} | TrainL={average_train_loss:.5f} | ValL={average_val_loss:.5f} | P={patience_counter}") # 减少输出

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                patience_counter = 0
                best_model_state = LWPT.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {run_identifier}, Epoch {epoch + 1}")
                    break
        else: # 无验证集
             best_model_state = LWPT.state_dict() # 保存最后模型

    # --- 保存模型 (修改文件名) ---
    if best_model_state is not None:
        model_filename = f"{col}_{mode}_{loss_type}_best_model.pth"
        model_path = os.path.join(output_dir, model_filename) # 直接使用传入的 output_dir
        try:
            torch.save(best_model_state, model_path)
            # print(f"成功保存模型: {model_path}") # 减少输出
        except Exception as e:
            print(f"保存模型失败 {model_path}: {e}")
    else:
        print(f"警告: {run_identifier} 无模型可保存。")

    column_training_time = time.time() - column_start_time
    print(f"完成: {run_identifier} | 耗时: {column_training_time:.2f}s | 最佳验证损失: {best_val_loss:.6f}")

    # --- 绘制并保存损失曲线 (修改文件名, DPI=600) ---
    if train_losses or val_losses:
        plt.figure(figsize=(10, 5)) # DPI 在 savefig 中设置
        if train_losses: plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'Train ({mode}, {loss_type})')
        if val_losses: plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'Val ({mode}, {loss_type})')
        plt.title(f'Loss: {col} ({mode}, {loss_type})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        loss_plot_filename = f"{col}_{mode}_{loss_type}_loss_curves.png"
        loss_plot_path = os.path.join(output_dir, loss_plot_filename)
        try:
            plt.savefig(loss_plot_path, bbox_inches='tight', dpi=600) # 设置 DPI
            # print(f"保存损失图: {loss_plot_path}") # 减少输出
        except Exception as e:
            print(f"保存损失图失败 {loss_plot_path}: {e}")
        plt.close()

    # --- 处理完整数据并保存 (修改文件名) ---
    if best_model_state is not None:
        # print(f"处理完整序列 ({run_identifier})...") # 减少输出
        LWPT.load_state_dict(best_model_state)
        LWPT.eval()
        full_original_normalized = []
        full_denoised_normalized = []
        with torch.no_grad():
            for i, x_batch_full in enumerate(full_loader):
                x_batch_full = x_batch_full.to(device)
                x_est_normalized = LWPT(x_batch_full)
                full_original_normalized.append(x_batch_full.cpu().numpy().flatten())
                full_denoised_normalized.append(x_est_normalized.cpu().numpy().flatten())

        full_original_normalized = np.concatenate(full_original_normalized)[:original_length]
        full_denoised_normalized = np.concatenate(full_denoised_normalized)[:original_length]
        full_original_denormalized = full_dataset.denormalize(full_original_normalized)
        full_denoised_denormalized = full_dataset.denormalize(full_denoised_normalized)

        # 保存降噪数据
        denoised_data_filename = f"{col}_{mode}_{loss_type}_denoised.csv"
        denoised_data_path = os.path.join(output_dir, denoised_data_filename)
        denoised_col_name = f'{col}_denoised_{mode}_{loss_type}'
        try:
            if time_column is not None:
                time_data = time_column.iloc[:original_length].reset_index(drop=True)
                min_len = min(len(time_data), len(full_denoised_denormalized))
                full_denoised_df = pd.DataFrame({time_column.name: time_data[:min_len],
                                                 f'{col}_original': full_original_denormalized[:min_len],
                                                 denoised_col_name: full_denoised_denormalized[:min_len]})
            else:
                 min_len = len(full_denoised_denormalized)
                 full_denoised_df = pd.DataFrame({f'{col}_original': full_original_denormalized[:min_len],
                                                 denoised_col_name: full_denoised_denormalized[:min_len]})
            full_denoised_df.to_csv(denoised_data_path, index=False)
            # print(f"保存数据: {denoised_data_path}") # 减少输出
        except Exception as e:
            print(f"保存数据失败 {denoised_data_path}: {e}")

        # --- 绘制重建对比图 (修改文件名, DPI=600) ---
        try:
             plt.figure(figsize=(15, 5)) # DPI 在 savefig 中设置
             plot_label = f'Denoised ({mode}, {loss_type})'
             if time_column is not None and len(time_column) >= original_length:
                 time_values = time_column.iloc[:original_length].values
                 plt.plot(time_values, full_original_denormalized, label='Original', linewidth=0.8, alpha=0.7, color='grey')
                 plt.plot(time_values, full_denoised_denormalized, label=plot_label, linewidth=1.0, color='red')
                 plt.xlabel('Time')
                 if len(time_values)>0: plt.xlim(time_values[0], time_values[-1])
             elif len(full_original_denormalized) > 0:
                 plt.plot(full_original_denormalized, label='Original', linewidth=0.8, alpha=0.7, color='grey')
                 plt.plot(full_denoised_denormalized, label=plot_label, linewidth=1.0, color='red')
                 plt.xlabel('Sample Index'); plt.xlim(0, original_length)

             plt.title(f'Original vs Denoised: {col} ({mode}, {loss_type})')
             plt.ylabel('Amplitude'); plt.legend(); plt.grid(True, alpha=0.6); plt.tight_layout()

             reconstruction_plot_filename = f"{col}_{mode}_{loss_type}_reconstruction.png"
             reconstruction_plot_path = os.path.join(output_dir, reconstruction_plot_filename)
             plt.savefig(reconstruction_plot_path, bbox_inches='tight', dpi=600) # 设置 DPI
             # print(f"保存重建图: {reconstruction_plot_path}") # 减少输出
        except Exception as e:
             print(f"绘制或保存重建图失败 {reconstruction_plot_path}: {e}")
        finally:
             plt.close()

    # --- 清理内存 (简化) ---
    del train_dataset, val_dataset, full_dataset, train_loader, val_loader, full_loader
    del LWPT, optimizer, loss_fn, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    # --- 配置参数 ---
    file_path = '/mnt/c/Users/shiki/Desktop/ocean engineering/proposed_method_padded_data.csv' # !! 确保路径正确 !!
    epochs = 1000
    patience = 150
    batch_size = 256
    Lambda = 0.1

    # --- 创建基础输出目录 (LWPTXRSY) ---
    try:
        dataset_dir = os.path.dirname(file_path)
        # **修改**: 直接定义目标输出文件夹名
        output_dir = os.path.join(dataset_dir, "LWPTXRSY")
        os.makedirs(output_dir, exist_ok=True) # 创建文件夹，如果不存在的话
        print(f"输出将保存到: {output_dir}")
    except Exception as e:
        print(f"创建输出目录 {output_dir} 失败: {e}")
        # 备选方案: 在当前工作目录创建
        output_dir = os.path.join(os.getcwd(), "LWPTXRSY")
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"使用备选输出目录: {output_dir}")
        except Exception as e_fallback:
            print(f"创建备选目录失败: {e_fallback}。程序退出。")
            sys.exit(1)

    # --- 加载数据 (与原版一致) ---
    try:
        full_data = pd.read_csv(file_path)
        print(f"成功读取数据: {file_path}, 形状: {full_data.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}"); sys.exit(1)

    # --- 确定原始数据长度 (与原版一致) ---
    time_col_name = full_data.columns[0]
    time_column_series = full_data[time_col_name]
    is_padding = time_column_series.isna() | (time_column_series == 0)
    padding_indices = np.where(is_padding)[0]
    original_length = padding_indices[0] if len(padding_indices) > 0 and padding_indices[0] != 0 else len(full_data)
    # 处理特殊情况 (开头就是填充 或 全部填充)
    if len(padding_indices) > 0 and padding_indices[0] == 0:
        if not is_padding.all():
            print(f"警告: 数据从索引0开始填充。")
            # 根据你的数据理解，决定 original_length 应该是0还是其他值
        else: # 全部是填充
            original_length = 0
    elif is_padding.all():
        original_length = 0
    print(f"检测到原始数据长度: {original_length}")
    data_length = len(full_data)
    print(f"完整数据长度: {data_length}")

    # --- 计算 Np (与原版一致) ---
    Np = find_nearest_power_of_two(data_length)
    print(f"使用块长度 Np: {Np}")

    # --- 获取要处理的数据列 (与原版一致) ---
    cols_to_process = full_data.columns.tolist()[1:]
    print(f"将处理的列: {cols_to_process}")

    # --- 定义要运行的消融组合 ---
    modes_to_run = ['Focus', 'NoFocus']
    loss_types_to_run = ['SmoothL1', 'L1']

    # --- **修改**: 循环处理每一列和每种消融组合 ---
    global_start_time = time.time()
    total_runs = len(cols_to_process) * len(modes_to_run) * len(loss_types_to_run)
    current_run = 0

    for col in cols_to_process:
        # 提取数据放外面，避免重复提取
        data_for_col = full_data[[time_col_name, col]].copy() if time_col_name in full_data.columns else full_data[[col]].copy()
        time_col_for_plot = data_for_col[time_col_name] if time_col_name in data_for_col.columns else None
        data_only_col = data_for_col[[col]].copy()

        for mode in modes_to_run:
            for loss_type in loss_types_to_run:
                current_run += 1
                print(f"\n>>> 运行 {current_run}/{total_runs}: Col={col}, Mode={mode}, Loss={loss_type}")
                try:
                    process_single_column(
                        col=col,
                        full_data_col=data_only_col,
                        original_length=original_length,
                        time_column=time_col_for_plot,
                        Np=Np,
                        epochs=epochs,
                        patience=patience,
                        batch_size=batch_size,
                        # **修改**: 传入统一的基础输出目录
                        output_dir=output_dir,
                        Lambda=Lambda,
                        # **修改**: 传入当前运行的模式和损失类型
                        mode=mode,
                        loss_type=loss_type
                    )
                    time.sleep(1) # 短暂暂停

                except Exception as e:
                    print(f"!!!!!! 运行 ({col}, {mode}, {loss_type}) 时出错: {e} !!!!!!")
                    traceback.print_exc()
                    print(f"!!!!!! 跳过此组合 !!!!!!")
                    continue # 继续下一个组合

    global_end_time = time.time()
    total_time_minutes = (global_end_time - global_start_time) / 60
    print(f"\n===== 所有运行完成。 =====")
    print(f"总耗时: {total_time_minutes:.2f} 分钟。")
    print(f"结果保存在: {output_dir}")