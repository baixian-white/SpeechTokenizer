import os
import json
import gc
import sys
from pathlib import Path
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import typing as tp
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NAS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 引入你自己的模块
from speechtokenizer.trainer.dataset import get_dataloader, audioDataset
from speechtokenizer.trainer.loss import mel_loss, recon_loss
from speechtokenizer.trainer.optimizer import get_optimizer
from speechtokenizer.modules import SConv1d, SConvTranspose1d, SLSTM

# ==========================
# 1️⃣ Snake 激活函数
# ==========================
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
    def forward(self, x):
        return snake(x, self.alpha)

# ==========================
# 2️⃣ SEANet 模块定义
# ==========================
class SEANetResnetBlock(nn.Module):
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 1], dilations: tp.List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), '卷积核数必须与膨胀率数一致'
        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            # 动态实例化激活函数，修复维度问题
            block += [
                act(**activation_params) if activation != 'Snake' else act(in_chs),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class SEANetEncoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2, bidirectional:bool = False):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios)) # Encoder使用反向顺序
        self.n_residual_layers = n_residual_layers
        
        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        mult = 1
        model: tp.List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      norm=norm, norm_params=norm_params,
                                      activation=activation, activation_params=activation_params,
                                      causal=causal, pad_mode=pad_mode,
                                      compress=compress, true_skip=true_skip)
                ]
            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConv1d(mult * n_filters, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]

        mult = mult * 2 if bidirectional else mult
        model += [
            act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
            SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class SEANetDecoder(nn.Module):
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[int] = [8, 5, 4, 2], activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2,
                 trim_right_ratio: float = 1.0, bidirectional:bool = False):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        self.n_residual_layers = n_residual_layers
        
        act = getattr(nn, activation) if activation != 'Snake' else Snake1d
        mult = int(2 ** len(self.ratios))
        
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm, bidirectional=bidirectional)]
            if bidirectional:
                mult = mult * 2

        for i, ratio in enumerate(self.ratios):
            model += [
                act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                 kernel_size=ratio * 2, stride=ratio,
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio),
            ]
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      activation=activation, activation_params=activation_params,
                                      norm=norm, norm_params=norm_params, causal=causal,
                                      pad_mode=pad_mode, compress=compress, true_skip=true_skip)
                ]
            mult //= 2

        model += [
            act(**activation_params) if activation != 'Snake' else act(mult * n_filters),
            SConv1d(mult * n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]

        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)

# =========================
# 3️⃣ NAS 搜索相关函数
# =========================

def define_search_space(trial):
    kernel_size = trial.suggest_int('kernel_size', 3, 7)
    n_residual_layers = trial.suggest_int('n_residual_layers', 1, 3)
    dilation_rate = trial.suggest_int('dilation_rate', 1, 4)
    compress = trial.suggest_int('compress', 1, 4)  
    lstm_units = trial.suggest_int('lstm_units', 64, 512)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    activation = trial.suggest_categorical('activation', ['ReLU', 'ELU', 'Snake', 'GELU'])

    # 下采样率组合搜索 (总倍数 320)
    ratios_options = [
        '[8, 5, 4, 2]', '[2, 4, 5, 8]', 
        '[4, 4, 5, 4]', '[5, 4, 4, 4]',
        '[10, 4, 4, 2]', '[2, 2, 8, 10]'
    ]
    ratios_choice = trial.suggest_categorical('ratios', ratios_options)

    return {
        'kernel_size': kernel_size,
        'ratios': eval(ratios_choice),
        'n_residual_layers': n_residual_layers,
        'dilation_rate': dilation_rate,
        'compress': compress,
        'lstm_units': lstm_units,
        'lstm_layers': lstm_layers,
        'bidirectional': bidirectional,
        'activation': activation
    }

HISTORY_FILE = ARTIFACTS_DIR / "nas_history.jsonl"
BEST_FILE = ARTIFACTS_DIR / "best_hyperparameters.json"

def save_trial_result(params, loss, trial_number):
    record = params.copy()
    record['loss'] = loss
    record['trial_id'] = trial_number
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def check_and_save_best(params, loss, trial_number):
    current_best_loss = float('inf')
    if BEST_FILE.exists():
        try:
            with open(BEST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                current_best_loss = data.get('loss', float('inf'))
        except:
            pass
    if loss < current_best_loss:
        print(f"\n🔥 [New Record] Trial {trial_number} beat old best {current_best_loss:.4f} -> {loss:.4f}!")
        best_data = params.copy()
        best_data['loss'] = loss
        best_data['trial_id'] = trial_number
        with open(BEST_FILE, "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=4)
        return True
    return False

def objective(trial):
    gc.collect()
    torch.cuda.empty_cache()

    params = define_search_space(trial)
    print(f"\n{'='*10} Trial {trial.number} {'='*10}")

    # ============================
    # 🔴 针对 RTX 4090 的性能优化
    # ============================
    target_batch_size = 16
    real_batch_size = 8  # 从 4 增加到 8，充分利用显存
    accumulation_steps = max(1, target_batch_size // real_batch_size)
    num_workers = 4      # 开启多线程加载，防止 GPU 等待 CPU

    mel_params = {
        "n_fft": 1024, "num_mels": 80, "sample_rate": 16000,
        "hop_size": 240, "win_size": 1024, "fmin": 0, "fmax": 8000
    }

    try:
        current_activation = params['activation']
        current_act_params = {'alpha': 1.0} if current_activation == 'ELU' else {}

        # 传入 ratios 参数
        encoder = SEANetEncoder(
            channels=1, dimension=1024, n_filters=32,
            n_residual_layers=params['n_residual_layers'], 
            ratios=params['ratios'], 
            activation=current_activation, activation_params=current_act_params,
            kernel_size=params['kernel_size'], dilation_base=params['dilation_rate'],
            lstm=params['lstm_layers'], bidirectional=params['bidirectional']
        )
        
        decoder = SEANetDecoder(
            channels=1, dimension=1024, n_filters=32,
            n_residual_layers=params['n_residual_layers'], 
            ratios=params['ratios'], 
            activation=current_activation, activation_params=current_act_params,
            kernel_size=params['kernel_size'], dilation_base=params['dilation_rate'],
            lstm=params['lstm_layers'], bidirectional=params['bidirectional']
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        decoder.to(device)

    except torch.cuda.OutOfMemoryError:
        print(f"⚠️ Trial {trial.number} Init OOM. Pruning.")
        gc.collect()
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()

    # 数据集加载
    train_files = PROJECT_ROOT / "data" / "SpeechPretrain" / "hubert_rep" / "LibriSpeech" / "train_files.txt"
    valid_files = PROJECT_ROOT / "data" / "SpeechPretrain" / "hubert_rep" / "LibriSpeech" / "valid_files.txt"
    segment_size = 48000 
    sample_rate = 16000 
    
    # 读取文件列表
    train_file_list = open(train_files, "r", encoding="utf-8").readlines()
    valid_file_list = open(valid_files, "r", encoding="utf-8").readlines()
    
    train_dataset = audioDataset(file_list=train_file_list, segment_size=segment_size, sample_rate=sample_rate)
    valid_dataset = audioDataset(file_list=valid_file_list, segment_size=segment_size, sample_rate=sample_rate, valid=True)

    # 🔴 开启 num_workers 和 pin_memory
    train_loader = get_dataloader(train_dataset, batch_size=real_batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    valid_loader = get_dataloader(valid_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    optim_g = get_optimizer(encoder.parameters(), lr=0.001)
    optim_d = get_optimizer(decoder.parameters(), lr=0.001)

    n_epochs = 1 
    
    try:
        # --- Train ---
        for epoch in range(n_epochs):
            encoder.train()
            decoder.train()
            pbar = tqdm(train_loader, desc=f"Trial {trial.number} [Train]", leave=False)
            
            optim_g.zero_grad()
            optim_d.zero_grad()
            
            for i, data in enumerate(pbar):
                inputs, targets = data
                if inputs.dim() == 2: inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)
                
                z = encoder(inputs)
                generated_audio = decoder(z)
                
                recon_loss_value = recon_loss(inputs, generated_audio)
                mel_loss_value = mel_loss(inputs, generated_audio, **mel_params)
                
                total_loss = (recon_loss_value + mel_loss_value) / accumulation_steps
                
                if torch.isnan(total_loss):
                    raise optuna.exceptions.TrialPruned()

                total_loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optim_g.step()
                    optim_d.step()
                    optim_g.zero_grad()
                    optim_d.zero_grad()
                
                loss_disp = total_loss.item() * accumulation_steps
                # 及时释放图和张量
                del z, generated_audio, total_loss, inputs
                pbar.set_postfix({'Loss': f"{loss_disp:.4f}"})

        # --- Validation ---
        encoder.eval()
        decoder.eval()
        total_valid_loss = 0.0
        valid_pbar = tqdm(valid_loader, desc=f"Trial {trial.number} [Valid]", leave=False)
        
        with torch.no_grad():
            for data in valid_pbar:
                inputs, targets = data
                if inputs.dim() == 2: inputs = inputs.unsqueeze(1)
                inputs = inputs.to(device)
                
                generated_audio = decoder(encoder(inputs))
                r_loss = recon_loss(inputs, generated_audio)
                m_loss = mel_loss(inputs, generated_audio, **mel_params)
                
                total_valid_loss += (r_loss + m_loss).item()
                valid_pbar.set_postfix({'Val': f"{(r_loss + m_loss).item():.4f}"})
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        pbar.close()
        valid_pbar.close()

        save_trial_result(params, avg_valid_loss, trial.number)
        check_and_save_best(params, avg_valid_loss, trial.number)
        
        print(f"✅ Trial {trial.number} Done. Loss: {avg_valid_loss:.4f}")
        return avg_valid_loss

    except torch.cuda.OutOfMemoryError:
        print(f"⚠️ Trial {trial.number} OOM. Pruning.")
        gc.collect()
        torch.cuda.empty_cache()
        save_trial_result(params, 9999.0, trial.number)
        raise optuna.exceptions.TrialPruned()
        
    finally:
        del encoder, decoder, optim_g, optim_d, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()

# =========================
# 执行优化 (并行版)
# =========================

if __name__ == "__main__":
    db_path = ARTIFACTS_DIR / "nas_search.db"
    storage_name = f"sqlite:///{db_path.resolve().as_posix()}"
    study_name = "sean_net_nas_study"

    print(f"Start NAS search using Database: {storage_name}")
    print(f"History file: {HISTORY_FILE}")
    print(f"Best config file: {BEST_FILE}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        load_if_exists=True 
    )
    
    study.optimize(objective, n_trials=100)

    print("\nOptimization Finished!")
    print(f"Best params found: {study.best_params}")
