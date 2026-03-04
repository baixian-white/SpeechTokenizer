# search_autoencoder.py
"""
脚本名称: search_autoencoder.py
功能描述: 
    这是 NAS (神经网络架构搜索) 的核心驱动脚本。
    它使用 Optuna 框架自动搜索 SpeechTokenizer (SEANet) 的最佳架构参数。

工作流程 (Objective Function):
    1. 采样 (Sample): 从定义好的搜索空间中选出一组参数 (如 Kernel Size, 算子类型)。
    2. 实例化 (Init): 根据参数构建 SEANet 模型。
    3. 门控 (Gating): 计算模型 FLOPs，如果超过阈值 (4G FLOPs) 直接跳过，不训练。
    4. 训练 (Train): 使用小规模数据集 (Proxy Dataset) 进行快速训练 (Few Epochs)。
    5. 评估 (Eval): 计算验证集 Loss 和 SI-SNR。
    6. 反馈 (Report): 将结果返回给 Optuna，指导下一次搜索。

输出:
    - seanet_nas_micro_v1.db: SQLite 数据库，存储所有搜索记录。
    - nas_records.csv: CSV 格式的详细记录，方便 Excel 查看。
    - debug_samples/: 最佳模型的重构音频和频谱图对比。

依赖库:
    pip install optuna thop filelock matplotlib
"""

import os
import gc
import json
import sys
from pathlib import Path
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import typing as tp
import numpy as np
import torchaudio
from tqdm import tqdm
from thop import profile
import csv
import matplotlib.pyplot as plt
import torchaudio.transforms as T

# 🛠️ [修复] 引入文件锁，防止多进程运行时写入 CSV 冲突
from filelock import FileLock 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NAS_DIR / "artifacts"
RESULTS_DIR = NAS_DIR / "架构搜索结果"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 假设 dataset/loss 在这些路径，如果报错请调整
from speechtokenizer.trainer.dataset import get_dataloader, audioDataset
from speechtokenizer.trainer.loss import mel_loss, recon_loss

# 🌟 引入刚写好的 SEANet
try:
    from .SeaNet import SEANet
except ImportError as e:
    try:
        from SeaNet import SEANet
    except ImportError:
        raise ImportError(f"❌ 找不到 SeaNet.py，请检查 nas 目录文件结构。\n详细错误: {e}")

# ==========================================
# 全局配置
# ==========================================
DB_PATH = ARTIFACTS_DIR / "seanet_nas_micro_v1.db"
DB_URL = f"sqlite:///{DB_PATH.resolve().as_posix()}"  # 数据库路径 (SQLite)
STUDY_NAME = "seanet_micro_search_v1"          # 实验名称
TOTAL_FLOPS_LIMIT = 4_000_000_000              # FLOPs 限制 (4G)，超过此限制的模型直接淘汰
TRAIN_SUBSET_PATH = ARTIFACTS_DIR / "train_subset_nas.txt"  # 代理数据集路径 (由 make_subset.py 生成)
GLOBAL_BEST_LOSS = float('inf')                # 记录全局最优 Loss
FIXED_TEST_AUDIO_PATH = PROJECT_ROOT / "samples" / "example_input.wav"  # 用于可视化的固定测试音频

# ==========================
# 辅助函数
# ==========================
def calculate_sisnr(ref, est):
    """计算 SI-SNR (尺度不变信噪比)，用于评估波形重构质量"""
    eps = 1e-8
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    est = est - torch.mean(est, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True) + eps
    projection = torch.sum(ref * est, dim=-1, keepdim=True) * ref / ref_energy
    noise = est - projection
    ratio = torch.sum(projection ** 2, dim=-1, keepdim=True) / (torch.sum(noise ** 2, dim=-1, keepdim=True) + eps)
    si_snr = 10 * torch.log10(ratio + eps)
    return si_snr.mean().item()

def save_comparison_plot(ref_wav, est_wav, trial_id, save_path):
    """绘制并保存 原始音频 vs 重构音频 的 Mel 频谱对比图"""
    # 确保 MelSpectrogram 在正确的设备上
    to_mel = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=240, n_mels=80).to(ref_wav.device)
    
    # 1. 计算 Spectrogram
    ref_spec_tensor = to_mel(ref_wav).log2()
    est_spec_tensor = to_mel(est_wav).log2()
    
    # 2. 【关键修正】使用 .squeeze() 去掉所有为 1 的维度
    # 结果会从 [1, 1, 80, 331] 变成 [80, 331]
    ref_spec = ref_spec_tensor.squeeze().cpu().detach().numpy()
    est_spec = est_spec_tensor.squeeze().cpu().detach().numpy()
    
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.imshow(ref_spec, aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Original Audio (Trial {trial_id})")
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 1, 2)
    plt.imshow(est_spec, aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Reconstructed Audio")
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 搜索空间定义
# ==========================================
def define_search_space(trial):
    """
    定义 Optuna 的超参数搜索空间。
    Trial 会根据这里定义的分布建议一组参数。
    """
    # 1. 宏观参数 (Macro): 影响整个模型的宽度和深度
    n_filters = trial.suggest_categorical('n_filters', [24, 32]) 
    lstm_layers = trial.suggest_categorical('lstm_layers', [1, 2])
    compress = trial.suggest_categorical('compress', [2, 4])
    activation = trial.suggest_categorical('activation', ['ELU', 'Snake'])
    
    # 2. 微观参数 (Micro): 每一层具体使用什么算子
    num_layers = 4 
    ops = []
    ses = []
    for i in range(num_layers):
        # 算子选择: 标准卷积 / 深度可分离 / 空洞 / 跳过
        op_name = trial.suggest_categorical(f'layer_{i}_op', ['std_k3', 'sep_k7', 'dil_k9', 'skip'])
        # SE 模块: 是否开启
        use_se = trial.suggest_categorical(f'layer_{i}_se', [True, False])
        ops.append(op_name)
        ses.append(use_se)

    return {
        'n_filters': n_filters,
        'dimension': 1024,
        'layer_ops_list': ops,
        'layer_se_list': ses,
        'lstm_layers': lstm_layers,
        'compress': compress,
        'activation': activation,
    }

# ==========================================
# 目标函数 (Search Loop)
# ==========================================
def objective(trial):
    # 清理内存，防止 OOM
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 获取本轮 Trial 的参数
    params = define_search_space(trial)

    try:
        # 2. 模型初始化
        model = SEANet(
            channels=1, 
            dimension=params['dimension'],
            n_filters=params['n_filters'],
            lstm=params['lstm_layers'],   
            activation=params['activation'],
            compress=params['compress'],
            layer_ops_list=params['layer_ops_list'],
            layer_se_list=params['layer_se_list']
        ).to(device)
        
        encoder = model.encoder
        decoder = model.decoder

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        # 如果参数组合非法导致崩溃，标记为 Pruned
        raise optuna.exceptions.TrialPruned()

    # --- 3. FLOPs 门控 (Speed Constraint) ---
    try:
        encoder.eval()
        decoder.eval()
        dummy_input = torch.randn(1, 1, 16000).to(device)
        # 使用 thop 计算计算量
        enc_flops, _ = profile(encoder, inputs=(dummy_input,), verbose=False)
        with torch.no_grad():
            z = encoder(dummy_input)
        dec_flops, _ = profile(decoder, inputs=(z,), verbose=False)
        total_flops = enc_flops + dec_flops
        
        # 将 FLOPs 记录到 Optuna 属性中，方便后续分析
        trial.set_user_attr("total_flops", total_flops)
        
        # 硬约束：如果模型太重，直接剪枝，不进行训练
        if total_flops > TOTAL_FLOPS_LIMIT:
            raise optuna.exceptions.TrialPruned()
            
    except Exception as e:
        if "TrialPruned" in str(e): raise e
        return float('inf')

    # --- 4. 训练准备 ---
    if not TRAIN_SUBSET_PATH.exists():
        print(f"❌ 找不到 {TRAIN_SUBSET_PATH}")
        return float('inf')
        
    try:
        # 加载小规模数据集
        with open(TRAIN_SUBSET_PATH, "r", encoding="utf-8") as f:
            file_list = f.readlines()
        train_dataset = audioDataset(file_list=file_list, segment_size=16000, sample_rate=16000)
        
        if len(train_dataset) == 0:
            print("❌ 数据集为空！")
            return float('inf')
            
        train_loader = get_dataloader(train_dataset, batch_size=16, num_workers=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # 搜索阶段 Epoch 不需太多，主要看收敛趋势
        n_epochs = 100 
        final_loss = float('inf')
        
        model.train()
        x = torch.zeros(1, 1, 16000).to(device)
        out = torch.zeros(1, 1, 16000).to(device)

        # --- 5. 训练循环 ---
        for epoch in range(n_epochs):
            total_loss = 0
            count = 0
            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep {epoch+1}", leave=False)
            
            for batch in pbar:
                x, _ = batch
                x = x.to(device).unsqueeze(1)
                
                out, _ = model(x)
                
                # 计算 Loss (重构 Loss + Mel 频谱 Loss)
                l_recon = recon_loss(x, out)
                l_mel = mel_loss(x, out, n_fft=1024, num_mels=80, sample_rate=16000, hop_size=240, win_size=1024, fmin=0, fmax=8000)
                loss = l_recon + l_mel
                
                # 检查 NaN
                if torch.isnan(loss):
                    print("❌ Loss is NaN")
                    raise optuna.exceptions.TrialPruned()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                pbar.set_postfix({'loss': loss.item()})
            
            # 计算平均 Loss
            if count > 0:
                avg_loss = total_loss / count
                final_loss = avg_loss
                # 向 Optuna 报告当前 Epoch 的 Loss
                trial.report(avg_loss, epoch)
            else:
                final_loss = float('inf')

            # --- 6. 自动剪枝 (Pruning) ---
            # 如果当前 Loss 远差于平均水平，Optuna 会提前终止训练，节省时间
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # --- 7. 结果记录与可视化 ---
        global GLOBAL_BEST_LOSS
        
        current_sisnr = calculate_sisnr(x, out)
        is_best_trial = False

        # 如果打破了全局最佳记录
        if final_loss < GLOBAL_BEST_LOSS:
            print(f"🎉 新纪录！Loss: {final_loss:.4f} | SI-SNR: {current_sisnr:.2f} dB")
            GLOBAL_BEST_LOSS = final_loss
            is_best_trial = True
            
            # 绘制并保存最佳样本对比图
            try:
                model.eval()
                FIXED_TEST_AUDIO_PATH.parent.mkdir(parents=True, exist_ok=True)
                
                if not FIXED_TEST_AUDIO_PATH.exists():
                    torchaudio.save(str(FIXED_TEST_AUDIO_PATH), x[0].cpu().detach(), 16000)
                
                wav_in, sr = torchaudio.load(str(FIXED_TEST_AUDIO_PATH))
                if sr != 16000:
                    wav_in = T.Resample(sr, 16000)(wav_in)
                
                wav_in = wav_in.to(device)
                if wav_in.shape[0] > 1: wav_in = wav_in[:1, :]
                wav_in = wav_in.unsqueeze(0)
                
                with torch.no_grad():
                    out_best, _ = model(wav_in)
                
                best_audio_path = RESULTS_DIR / f"best_audio_trial_{trial.number}.wav"
                best_spec_path = RESULTS_DIR / f"best_spec_trial_{trial.number}.png"
                torchaudio.save(str(best_audio_path), out_best.squeeze(0).cpu().detach(), 16000)
                save_comparison_plot(wav_in, out_best, trial.number, str(best_spec_path))
            except Exception as e_plot:
                print(f"⚠️ 保存图片/音频失败，但训练继续: {e_plot}")

        # --- 8. 写入 CSV 记录 ---
        csv_file = ARTIFACTS_DIR / "nas_records.csv"
        lock_file = ARTIFACTS_DIR / "nas_records.csv.lock"
        
        try:
            # 使用文件锁，确保多进程并行搜索时不冲突
            with FileLock(str(lock_file), timeout=10):
                file_exists = csv_file.exists()
                with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists or csv_file.stat().st_size == 0:
                        writer.writerow(['Trial_ID', 'Loss', 'SI-SNR(dB)', 'FLOPs(G)', 'Filters', 'LSTM', 'Status'])
                    
                    writer.writerow([
                        trial.number, 
                        f"{final_loss:.4f}", 
                        f"{current_sisnr:.2f}", 
                        f"{total_flops/1e9:.3f}", 
                        params['n_filters'], 
                        params['lstm_layers'], 
                        "Best" if is_best_trial else "Normal"
                    ])
        except Exception as e_csv:
            print(f"⚠️ CSV写入失败: {e_csv}")   

    except Exception as e:
        if "TrialPruned" in str(e): raise e
        print(f"🔥 Critical Training Err: {e}")
        return float('inf')

    # 返回验证集 Loss，作为 Optuna 的优化目标 (minimize)
    return final_loss

if __name__ == "__main__":
    # 使用 TPE (Tree-structured Parzen Estimator) 贝叶斯优化算法
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    
    # 创建 Study，指向 SQLite 数据库实现断点续搜
    study = optuna.create_study(
        study_name=STUDY_NAME, 
        storage=DB_URL, 
        direction='minimize', 
        sampler=sampler, 
        load_if_exists=True
    )
    
    # 开始搜索 (n_trials 决定了总共尝试多少种组合)
    study.optimize(objective, n_trials=300)
