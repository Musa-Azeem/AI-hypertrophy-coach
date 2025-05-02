import torch
from torch import nn
import torch.nn.functional as F
import torch.quantization
import numpy as np
import pandas as pd
import copy
import warnings
from collections import deque
# Added imports for data loading/splitting
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
# --- Added sklearn imports for accuracy metrics ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import json # For loading config if needed separately
# from torch.profiler import profile, record_function, ProfilerActivity # Keep if needed for profiling later


# --- Constants ---
# Defaults - will be primarily determined by loaded config or dataset logic
HZ = 100
in_channels = 6
winsize = 256
stride = 2 # Stride used in IMUDataset __getitem__ for selecting data slice start
n_LSTM_windows = 32
LSTM_stride = 64
# full_winsize will be calculated later based on loaded config

# --- Helper Functions ---
# (Keep moving_average and get_time_in_reps as defined in the previous answer)
def moving_average(data, window_size=15):
    """Applies moving average smoothing."""
    if isinstance(data, torch.Tensor): data_np = data.cpu().numpy()
    else: data_np = data # Assume numpy already
    X_smooth = np.zeros_like(data_np); C = data_np.shape[0]
    for i in range(C):
        channel = data_np[i]
        if len(channel) >= window_size: X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
        else: X_smooth[i] = channel
    return torch.from_numpy(X_smooth).to(torch.float32)

def get_time_in_reps(preds, stride, winsize, full_winsize):
    """
    Calculates time within rep segments based on model predictions.
    (Using the version from user script - requires careful validation)
    Args:
        preds (Tensor): Model predictions, expected shape (N, T_in, L_pred), detached on CPU.
        stride (int): The stride used for unfolding (LSTM_stride in this context).
        winsize (int): The window size used for unfolding.
        full_winsize (int): The total length of the sequence used for normalization.
    Returns:
        Tensor: Processed time_in_rep features, shape (N, T_new, winsize).
    """
    preds_cpu = preds.detach().cpu() # Ensure CPU
    N, T_in, L_pred = preds_cpu.shape

    time_in_rep_batch = []
    for pred_single_batch in preds_cpu: # Iterate through batch
        # Consolidate predictions across time dimension T_in
        consolidatedi = []
        for i, predi in enumerate(pred_single_batch):
            upper = stride if i < T_in - 1 else L_pred # Use stride (LSTM_stride)
            consolidatedi.append(predi[:upper])

        try:
            consolidatedi_tensor = torch.cat(consolidatedi, axis=0) # Shape approx (T_in * stride)
        except Exception as e:
            print(f"Error during torch.cat in get_time_in_reps: {e}")
            T_new = (L_pred - winsize) // stride + 1 if L_pred >= winsize else 0
            return torch.zeros(N, T_new, winsize) # Return empty tensor if error

        # Perform numpy operations
        try:
            consolidatedi_np = consolidatedi_tensor.numpy().astype(np.float64)
            if len(consolidatedi_np) >= 100:
                 smoothed_np = np.convolve(consolidatedi_np, np.ones(100)/100.0, mode='same').round()
            else:
                 warnings.warn(f"Sequence length {len(consolidatedi_np)} < 100, skipping convolution smoothing.")
                 smoothed_np = consolidatedi_np.round()
            consolidatedi = torch.from_numpy(smoothed_np).to(torch.float32)
        except Exception as e:
            print(f"Error during numpy ops in get_time_in_reps: {e}")
            consolidatedi = consolidatedi_tensor.round().to(torch.float32) # Fallback

        # Calculate time features based on starts/ends
        time_in_repi = torch.zeros_like(consolidatedi).to(torch.float32)
        end_repsi = []
        if consolidatedi.sum() > 0:
            diff = np.diff(consolidatedi.numpy())
            starts = np.where(diff > 0)[0] + 1; ends = np.where(diff < 0)[0] + 1

            # Handle edge cases carefully
            if len(consolidatedi) > 0 and consolidatedi[0] > 0 and (len(starts) == 0 or (len(ends) > 0 and starts[0] > ends[0])): starts = np.insert(starts, 0, 0)
            if len(consolidatedi) > 0 and consolidatedi[-1] > 0 and (len(ends) == 0 or (len(starts) > 0 and ends[-1] < starts[-1])): ends = np.append(ends, len(consolidatedi))

            if len(starts) > len(ends): ends = np.append(ends, len(consolidatedi))
            elif len(ends) > len(starts): starts = np.insert(starts, 0, 0)

            min_len = min(len(starts), len(ends))
            starts = starts[:min_len]; ends = ends[:min_len]
            valid_indices = [i for i, (s, e) in enumerate(zip(starts, ends)) if e > s]
            starts = starts[valid_indices]; ends = ends[valid_indices]

            for start, end in zip(starts, ends): end_repsi.append((start + end) // 2)

            last_end = 0
            for i in range(len(end_repsi)):
                start_interval = last_end; end_interval = end_repsi[i]
                if end_interval > start_interval:
                    # --- FIX HERE: Remove the redundant check ---
                    norm_factor = float(full_winsize) if full_winsize > 0 else 1.0
                    # --- END FIX ---
                    # Check removed: if norm_factor == 0: warnings.warn("full_winsize is zero"); norm_factor=1.0
                    duration = max(0.0, (end_interval - start_interval) / norm_factor)
                    end_assign = min(end_interval, len(time_in_repi));
                    if start_interval < end_assign: time_in_repi[start_interval:end_assign] = duration
                last_end = end_interval
        time_in_rep_batch.append(time_in_repi) # Append the processed tensor

    # Pad sequences in the batch
    max_len = max(len(t) for t in time_in_rep_batch) if time_in_rep_batch else 0
    padded_time_in_rep = []
    for t in time_in_rep_batch: padding_len = max_len - len(t); padded_t = F.pad(t, (0, padding_len)); padded_time_in_rep.append(padded_t)

    if not padded_time_in_rep:
        T_new = (max_len - winsize) // stride + 1 if max_len >= winsize else 0
        return torch.zeros((N, T_new, winsize), dtype=torch.float32)

    time_in_rep_stacked = torch.stack(padded_time_in_rep, axis=0) # Shape (N, max_len)

    # Rewindow / Unfold along the time dimension (dim 1)
    if time_in_rep_stacked.size(1) >= winsize:
        time_in_rep_unfolded = time_in_rep_stacked.unfold(1, winsize, stride) # Use stride (LSTM_stride)
    else:
        T_new = 0; warnings.warn(f"Seq len {time_in_rep_stacked.size(1)} < {winsize}. Cannot unfold."); return torch.zeros((N, T_new, winsize), dtype=torch.float32)

    # Expected output shape (N, T_new, winsize)
    return time_in_rep_unfolded

# --- Data Loading Class (from notebook) ---
class IMUDataset(Dataset):
    # (Keep implementation from previous answer)
    def __init__(self, df, winsize, stride, full_winsize, n_LSTM_windows, LSTM_stride, transform=None, aug=False, extra_pad=False):
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'rir']
        if not all(col in df.columns for col in required_cols): raise ValueError(f"DataFrame missing columns: {required_cols}")
        self.X = torch.from_numpy(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values)
        self.y = torch.from_numpy(df['rir'].fillna(0).values).to(torch.float32)
        if transform is not None: self.X = transform(self.X)
        self.X = self.X.transpose(0, 1).to(torch.float32) # Shape (C, L)

        self.winsize = winsize; self.stride = stride; self.full_winsize = full_winsize
        self.n_LSTM_windows = n_LSTM_windows; self.LSTM_stride = LSTM_stride; self.aug = aug

        self.len = (self.X.shape[1] - self.full_winsize) // self.stride + 1
        if self.len <= 0:
            pad = self.full_winsize - self.X.shape[1]
            if pad > 0: self.X = torch.cat([self.X, torch.zeros(self.X.shape[0], pad)], dim=1); self.y = torch.cat([self.y, torch.zeros(pad)])
            self.len = 1 if self.X.shape[1] >= self.full_winsize else 0

        if extra_pad:
            pad = (self.winsize // 2) * self.stride
            if pad > 0: self.X = torch.cat([self.X, torch.zeros(self.X.shape[0], pad)], dim=1); self.y = torch.cat([self.y, torch.zeros(pad)])
            self.len = (self.X.shape[1] - self.full_winsize) // self.stride + 1

        if self.len <= 0: self.len = 0

    def __len__(self): return self.len

    def __getitem__(self, i):
        if i >= self.len: raise IndexError("Index out of bounds")
        start = i * self.stride; end = start + self.full_winsize
        if end > self.X.shape[1]: end = self.X.shape[1]; start = max(0, end - self.full_winsize)

        X_slice = self.X[:, start:end]; y_slice = self.y[start:end]
        X_smooth = moving_average(X_slice)

        X_unfolded = X_smooth.unfold(1, self.winsize, self.LSTM_stride) # (C, T, winsize)
        X_out = X_unfolded.permute(1, 0, 2) # (T, C, winsize)
        y_unfolded = y_slice.unfold(0, self.winsize, self.LSTM_stride) # (T, winsize)
        y_out = (y_unfolded.mean(dim=1) < 2.5).float().unsqueeze(1) # (T, 1)

        current_T = X_out.shape[0]
        if current_T != self.n_LSTM_windows:
            warnings.warn(f"Item {i} shape T={current_T} != {self.n_LSTM_windows}. Padding/truncating T dim.")
            if current_T > self.n_LSTM_windows: X_out = X_out[:self.n_LSTM_windows, :, :]; y_out = y_out[:self.n_LSTM_windows, :]
            else:
                pad_T = self.n_LSTM_windows - current_T
                X_out = F.pad(X_out, (0, 0, 0, 0, 0, pad_T)); y_out = F.pad(y_out, (0, 0, 0, pad_T))

        return X_out, y_out

# --- Quantization-Ready Model Definitions ---
class ResBlock(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0,):
        super().__init__(); self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False); self.bn = nn.BatchNorm1d(out_channels); self.relu = nn.ReLU(); self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels or stride > 1: self.skip = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        else: self.skip = nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x): identity = self.skip(x); out = self.conv(x); out = self.bn(out); out = self.relu(out); out = self.dropout(out); out = self.skip_add.add(out, identity); return out

class DepthBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, kernel_size=3, downsample_stride=2, dropout=0.0):
        super().__init__(); self.blocks = nn.Sequential(*[ ResBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=downsample_stride if i == 0 else 1, dropout=dropout) for i in range(depth)])
    def forward(self, x): return self.blocks(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__(); self.width = config.get('width', []); self.depth = config.get('depth', []); self.stem_out_c = config.get('stem_out_c', 32); self.stem_kernel = config.get('stem_kernel', 7); self.dropout = config.get('dropout', 0.1); _in_channels = in_channels
        if len(self.width) != len(self.depth): raise ValueError('Encoder depth/width mismatch');
        self.conv_out_channels = self.stem_out_c if not self.width else self.width[-1]
        encoder_layers = [ResBlock(_in_channels, self.stem_out_c, kernel_size=self.stem_kernel, stride=2, dropout=self.dropout)]; current_channels = self.stem_out_c
        for i in range(len(self.width)): encoder_layers.append(DepthBlock(depth=self.depth[i], in_channels=current_channels, out_channels=self.width[i], dropout=self.dropout)); current_channels = self.width[i]
        self.encoder = nn.Sequential(*encoder_layers)
    def forward(self, x): return self.encoder(x)

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config; self.encoder = Encoder(config); self.ap = nn.AdaptiveAvgPool1d(1); _winsize = winsize
        self.fc = nn.Linear(self.encoder.conv_out_channels, _winsize, bias=True); self.quant_func = nn.quantized.FloatFunctional()
    def forward(self, x): emb = self.encoder(x); x = self.ap(emb); x = x.squeeze(-1); seg_logits = self.fc(x); return emb, seg_logits

class LSTMNet(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config; self.in_channels = in_channels
        lstm_cfg = config.get('lstm_config', {}); self.hidden_size = lstm_cfg.get('hidden_size', 256); self.skip_channels = lstm_cfg.get('skip_channels', 32); self.num_layers = lstm_cfg.get('num_layers', 2); dropout = lstm_cfg.get('dropout', 0.0); self.linear_hidden_size = lstm_cfg.get('linear_hidden_size', 0)
        self.full_winsize = LSTM_stride * (n_LSTM_windows - 1) + winsize
        config['full_winsize'] = self.full_winsize
        if dropout > 0: warnings.warn("LSTM dropout must be 0 for dynamic quantization."); dropout = 0.0
        self.quant = torch.quantization.QuantStub(); self.dequant_logits = torch.quantization.DeQuantStub(); self.quant_time_rep = torch.quantization.QuantStub(); self.dequant_lstm_input = torch.quantization.DeQuantStub(); self.quant_lstm_output = torch.quantization.QuantStub(); self.dequant_final = torch.quantization.DeQuantStub()
        self.encoder = ConvNet(config)
        self.conv_skip_modules = nn.Sequential( nn.Conv1d(self.in_channels, self.skip_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm1d(self.skip_channels), nn.ReLU())
        lstm_proj_in_features = self.encoder.encoder.conv_out_channels + self.skip_channels + winsize
        self.lstm_proj = nn.Linear(lstm_proj_in_features, self.hidden_size, bias=True)
        if self.linear_hidden_size == 0: self.fc = nn.Linear(self.hidden_size, 1, bias=True)
        else: self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.linear_hidden_size, bias=True), nn.ReLU(), nn.Linear(self.linear_hidden_size, 1, bias=True))
        self.lstm = nn.LSTM( input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=0.0)
        self.ap = nn.AdaptiveAvgPool1d(1); self.quant_func = nn.quantized.FloatFunctional()

    def forward(self, x):
        N, T, C, L = x.shape; x_orig_device = x.device; x = self.quant(x); x_reshaped = x.view(N*T, C, L)
        x_seg_emb_q, seg_logits_q = self.encoder(x_reshaped)
        seg_logits_q_reshaped = seg_logits_q.view(N, T, -1); seg_logits_float = self.dequant_logits(seg_logits_q_reshaped)
        seg_preds_float = torch.sigmoid(seg_logits_float).round()
        _LSTM_stride = LSTM_stride; _winsize = winsize
        time_in_rep_float = get_time_in_reps(seg_preds_float, _LSTM_stride, _winsize, self.full_winsize)
        time_in_rep_float = time_in_rep_float.to(x_orig_device); time_in_rep_q = self.quant_time_rep(time_in_rep_float)
        x_skip_q = self.conv_skip_modules(x_reshaped); x_skip_q = self.ap(x_skip_q); x_skip_q = x_skip_q.squeeze(-1); x_skip_q = x_skip_q.view(N, T, -1)
        x_seg_q = self.ap(x_seg_emb_q); x_seg_q = x_seg_q.squeeze(-1); x_seg_q = x_seg_q.view(N, T, -1)
        current_T = min(x_seg_q.shape[1], x_skip_q.shape[1], time_in_rep_q.shape[1])
        if current_T == 0 and T > 0: warnings.warn("Zero sequence length after processing."); return torch.zeros((N, T, 1), device=x_orig_device)
        x_seg_q = x_seg_q[:, :current_T, :]; x_skip_q = x_skip_q[:, :current_T, :]; time_in_rep_q = time_in_rep_q[:, :current_T, :]
        if time_in_rep_q.shape[-1] != _winsize and current_T > 0:
            warnings.warn(f"time_in_rep last dim {time_in_rep_q.shape[-1]} != winsize {_winsize}. Adjusting.")
            if time_in_rep_q.shape[-1] > _winsize: time_in_rep_q = time_in_rep_q[:, :, :_winsize]
            else: padding_needed = _winsize - time_in_rep_q.shape[-1]; time_in_rep_q = F.pad(time_in_rep_q, (0, padding_needed))
        lstm_input_features_q = self.quant_func.cat([x_seg_q, x_skip_q, time_in_rep_q], dim=2)
        lstm_input_q = self.lstm_proj(lstm_input_features_q); lstm_input_float = self.dequant_lstm_input(lstm_input_q)
        o_float, (h_float, c_float) = self.lstm(lstm_input_float); o_q = self.quant_lstm_output(o_float)
        output_q = self.fc(o_q); output_float = self.dequant_final(output_q); return output_float

    def fuse_model(self):
        print("Fusing modules..."); self.eval()
        def _fuse_recursive(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, nn.LSTM): continue
                if isinstance(child_module, ResBlock):
                    if hasattr(child_module, 'conv') and hasattr(child_module, 'bn') and hasattr(child_module, 'relu'): torch.quantization.fuse_modules(child_module, ['conv', 'bn', 'relu'], inplace=True)
                    if isinstance(child_module.skip, nn.Sequential) and len(child_module.skip) == 2 and isinstance(child_module.skip[0], nn.Conv1d) and isinstance(child_module.skip[1], nn.BatchNorm1d): torch.quantization.fuse_modules(child_module.skip, ['0', '1'], inplace=True)
                elif isinstance(child_module, nn.Sequential):
                    if name == 'conv_skip_modules' and len(child_module) == 3 and isinstance(child_module[0], nn.Conv1d) and isinstance(child_module[1], nn.BatchNorm1d) and isinstance(child_module[2], nn.ReLU): torch.quantization.fuse_modules(child_module, ['0', '1', '2'], inplace=True)
                    elif name == 'fc' and len(child_module) >= 2 and isinstance(child_module[0], nn.Linear) and isinstance(child_module[1], nn.ReLU): torch.quantization.fuse_modules(child_module, ['0', '1'], inplace=True)
                    else: _fuse_recursive(child_module)
                elif hasattr(child_module, 'named_children') and list(child_module.named_children()):
                    if not isinstance(child_module, nn.LSTM): _fuse_recursive(child_module)
        _fuse_recursive(self); print("Fusion attempt complete.")

# --- Quantization Function ---
def quantize_hybrid_static_dynamic(model_fp32, calibration_dataloader, device):
    """ (Keep implementation from previous answer) """
    model_hybrid = copy.deepcopy(model_fp32); model_hybrid.eval(); model_hybrid.to(device)
    print("--- Starting Static Quantization (Conv1d, Linear) ---"); model_hybrid.fuse_model()
    static_backend = "fbgemm" if str(device) == 'cpu' else "qnnpack"
    if str(device) == 'cuda': warnings.warn("Static PTQ CPU backend."); static_backend = "qnnpack"
    static_qconfig = torch.quantization.get_default_qconfig(static_backend)
    print(f"Assigning static qconfig globally and disabling for LSTM..."); model_hybrid.qconfig = static_qconfig
    for module_name, module in model_hybrid.named_modules():
        if isinstance(module, nn.LSTM): print(f"  Disabling qconfig for: {module_name}"); module.qconfig = None
    print("Preparing model for static quantization..."); model_prepared = torch.quantization.prepare(model_hybrid, inplace=False); model_prepared.eval()
    print(f"Running static calibration on {device}..."); batch_count = 0
    with torch.no_grad():
        for inputs, _ in calibration_dataloader:
            inputs = inputs.to(device); expected_T = n_LSTM_windows; expected_C = in_channels; expected_L = winsize
            if inputs.dim() != 4 or inputs.shape[1] != expected_T or inputs.shape[2] != expected_C or inputs.shape[3] != expected_L: warnings.warn(f"Calib shape mismatch: {inputs.shape}"); continue
            try: model_prepared(inputs); batch_count+=1
            except Exception as e: print(f"Error during calibration: {e}"); raise e
    print(f"Static calibration complete ({batch_count} batches)."); print("Converting static part..."); model_prepared.to('cpu'); model_static_quantized = torch.quantization.convert(model_prepared, inplace=False); model_static_quantized.eval()
    print("--- Static Quantization Complete ---"); print("\n--- Starting Dynamic Quantization (LSTM) ---"); dynamic_qconfig_spec = {nn.LSTM}; print(f"Applying dynamic quantization...")
    model_static_quantized.to('cpu'); model_final_hybrid = torch.quantization.quantize_dynamic(model=model_static_quantized, qconfig_spec=dynamic_qconfig_spec, dtype=torch.qint8, inplace=True); model_final_hybrid.eval()
    lstm_dynamic = any(isinstance(module, torch.nn.quantized.dynamic.LSTM) for module in model_final_hybrid.modules());
    if lstm_dynamic: print("Dynamic LSTM quantization successful.")
    else: warnings.warn("Dynamic LSTM quantization failed.")
    print("--- Dynamic Quantization Complete ---"); return model_final_hybrid

# --- Evaluation Function (Modified for Accuracy & Stride Mimic) ---
def evaluate_model_mimic_live(model, val_data_label_pairs, device, description="Model"):
    """
    Evaluates model mimicking the live inference data flow (inferring approx every LSTM_stride samples),
    calculates latency and accuracy.
    Args:
        model: The model to evaluate (FP32 or Quantized).
        val_data_label_pairs: A list of tuples, where each tuple is (session_data_tensor, session_rir_tensor).
                              Data tensor shape (C, L), RIR tensor shape (L,). Data should be normalized.
        device: The device to run evaluation on.
        description: A string description for printing.
    """
    model.eval()
    model.to(device)
    latencies = []
    inference_count = 0
    all_preds = []      # Store predictions (0 or 1)
    all_labels = []     # Store true labels (0 or 1)

    print(f"\n--- Evaluating {description} on {device} (Mimicking Live Inference, Stride={LSTM_stride}) ---")

    if not val_data_label_pairs:
        print("No validation data sessions provided.")
        return

    with torch.no_grad():
        # --- Loop through all provided sessions ---
        for session_idx, (session_data, session_rir) in enumerate(val_data_label_pairs):
            print(f"  Processing Session {session_idx+1}/{len(val_data_label_pairs)}...")
            # Ensure data is float tensor
            if not isinstance(session_data, torch.Tensor): session_data = torch.tensor(session_data, dtype=torch.float32)
            if not isinstance(session_rir, torch.Tensor): session_rir = torch.tensor(session_rir, dtype=torch.float32)
            session_data = session_data.to(torch.float32)
            session_rir = session_rir.to(torch.float32)

            C, session_L = session_data.shape
            if C != in_channels: warnings.warn(f"Session {session_idx} channel count ({C}), skipping."); continue
            if session_rir.shape[0] != session_L: warnings.warn(f"Session {session_idx} data/rir length mismatch, skipping."); continue
            min_len_for_inference = winsize + (n_LSTM_windows - 1)
            if session_L < min_len_for_inference: warnings.warn(f"Session {session_idx} too short ({session_L} < {min_len_for_inference}), skipping."); continue

            # Pre-apply moving average
            session_data_smooth = moving_average(session_data)

            input_windows = deque(maxlen=n_LSTM_windows)
            current_window_samples = deque(maxlen=winsize)
            current_window_rir = deque(maxlen=winsize)
            next_inference_allowed_at_idx = -1

            # Simulate sample arrival
            for k in range(session_L):
                current_window_samples.append(session_data_smooth[:, k])
                current_window_rir.append(session_rir[k])

                if len(current_window_samples) == winsize:
                    window_tensor = torch.stack(list(current_window_samples), dim=0).transpose(0, 1)
                    rir_window_tensor = torch.stack(list(current_window_rir), dim=0)
                    true_label = (rir_window_tensor.mean() < 2.5).float().item()
                    input_windows.append(window_tensor)

                    if len(input_windows) == n_LSTM_windows:
                        if k >= next_inference_allowed_at_idx:
                            model_input = torch.stack(list(input_windows), dim=0).to(device)
                            model_input = model_input.unsqueeze(0)

                            start_time = time.time(); outputs = model(model_input); end_time = time.time()

                            if outputs.dim() == 3 and outputs.shape[1] > 0: pred_logit = outputs[:, -1, 0]
                            elif outputs.dim() == 2: pred_logit = outputs[:, 0]
                            else: warnings.warn(f"Unexpected model out shape: {outputs.shape}."); pred_logit = torch.tensor([0.0], device=outputs.device)
                            prediction = torch.sigmoid(pred_logit).round().item()

                            latencies.append((end_time - start_time) * 1000)
                            all_preds.append(prediction)
                            all_labels.append(true_label)
                            inference_count += 1
                            next_inference_allowed_at_idx = k + LSTM_stride

            print(f"  Session {session_idx+1} finished processing.")
            # --- End of Session Loop ---

    print(f"--- Evaluation Loop Finished ({inference_count} inferences run across all sessions) ---")

    # --- Calculate and Print Overall Metrics ---
    if not latencies: print("No inference calls triggered during evaluation."); return

    # Latency
    avg_latency = np.mean(latencies); p95_latency = np.percentile(latencies, 95)
    print(f"\nOverall Latency Metrics:")
    print(f"Processed {inference_count} inference calls across all sessions.")
    print(f"Average Latency per call: {avg_latency:.3f} ms")
    print(f"P95 Latency per call:   {p95_latency:.3f} ms")

    # Accuracy Metrics
    if all_labels and all_preds and len(all_labels) == inference_count:
        print(f"\nOverall Accuracy Metrics (based on {inference_count} predictions):")
        all_labels_np = np.array(all_labels); all_preds_np = np.array(all_preds)
        accuracy = accuracy_score(all_labels_np, all_preds_np)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels_np, all_preds_np, average='binary', zero_division=0)
        print(f"Accuracy:  {accuracy:.4f}"); print(f"Precision: {precision:.4f}"); print(f"Recall:    {recall:.4f}"); print(f"F1-Score:  {f1:.4f}")
        counts = np.bincount(all_labels_np.astype(int), minlength=2); print(f"Support (True Labels): Class 0: {counts[0]}, Class 1: {counts[1]}")
    elif inference_count > 0: print("\nOverall Accuracy Metrics: Could not calculate (label/prediction mismatch).")

    print(f"--- Evaluation Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration and Model Loading ---
    saved_model_path = 'best_model-3binary-77.pth' # Path to the saved FLOAT model
    data_csv_path = 'data/data.csv' # Path to the data CSV
    device = torch.device("cpu") # Use CPU

    print(f"Loading model components from: {saved_model_path}")
    try:
        loaded_data = torch.load(saved_model_path, map_location=device)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            weights, config = loaded_data
            print("Successfully loaded weights and config.")
            if 'lstm_config' in config and config['lstm_config'].get('dropout', 0) > 0: config['lstm_config']['dropout'] = 0.0
            _n_lstm_windows = n_LSTM_windows # Use global constant
            config['full_winsize'] = LSTM_stride * (_n_lstm_windows - 1) + winsize
        else: raise TypeError("Loaded file is not a tuple of (weights, config)")
    except FileNotFoundError: print(f"Error: Model file not found at {saved_model_path}"); exit()
    except Exception as e: print(f"Error loading model file: {e}"); exit()

    print("Instantiating quantization-ready float model...")
    model_fp32 = LSTMNet(config)

    print("Loading state dict with strict=False...")
    missing_keys, unexpected_keys = model_fp32.load_state_dict(weights, strict=False)
    print(f"Missing keys count: {len(missing_keys)}")
    print(f"Unexpected keys count: {len(unexpected_keys)}")
    model_fp32.eval()

    # --- Load Real Data ---
    print(f"\nLoading data from: {data_csv_path}")
    try:
        df = pd.read_csv(data_csv_path)
        required_data_cols = ['session_id', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'rir', 'failure']
        if not all(col in df.columns for col in required_data_cols): print(f"Warning: CSV missing cols"); exit()
    except FileNotFoundError: print(f"Error: Data file not found at {data_csv_path}"); exit()
    except Exception as e: print(f"Error reading data CSV: {e}"); exit()

    # --- Preprocessing and Splitting ---
    print("Preprocessing data and splitting sessions...")
    session_ids = df['session_id'].unique()
    valid_session_ids = [sid for sid in session_ids if df[df.session_id == sid].failure.sum() > 0]
    if not valid_session_ids: print("Error: No sessions with failures found."); exit()
    try:
        train_ids, val_ids = train_test_split(valid_session_ids, test_size=0.2, random_state=42)
    except ValueError as e: print(f"Error split: {e}"); train_ids = []; val_ids = valid_session_ids
    print(f"Using {len(val_ids)} sessions for validation/calibration/testing.")
    if not val_ids: print("Error: No validation sessions available."); exit()

    # Normalize data
    print("Normalizing data...")
    df[['acc_x', 'acc_y', 'acc_z']] = (df[['acc_x', 'acc_y', 'acc_z']] / 2.0).clip(-1, 1)
    df[['gyr_x', 'gyr_y', 'gyr_z']] = (df[['gyr_x', 'gyr_y', 'gyr_z']] / 250.0).clip(-1, 1)

    # --- Create Full Validation Dataset (for Calibration) ---
    print("Creating validation dataset for calibration...")
    _stride = stride
    _full_winsize = config['full_winsize']
    val_dataset_list = []
    for session_id in val_ids:
         session_df = df[df['session_id'] == session_id]
         ds = IMUDataset(session_df, winsize, _stride, _full_winsize, n_LSTM_windows, LSTM_stride, transform=None, aug=False, extra_pad=True)
         if len(ds) > 0: val_dataset_list.append(ds)
         else: print(f"Warning: Session {session_id} produced no valid windows for calibration dataset.")
    if not val_dataset_list: print("Error: No calibration data windows created."); exit()
    val_dataset_full = ConcatDataset(val_dataset_list)
    print(f"Full validation dataset size (for calibration): {len(val_dataset_full)}")

    # --- Prepare ALL validation data for evaluate_model_mimic_live ---
    print("Extracting raw data & label tensors for mimic evaluation (ALL validation sessions)...")
    validation_session_data_label_pairs = [] # List of tuples (data_tensor, rir_tensor)
    # --- CHANGE: Loop through ALL val_ids ---
    for session_id in val_ids:
        print(f"  Preparing session_id: {session_id} for evaluation...")
        session_df = df[df['session_id'] == session_id].copy()
        session_data = torch.from_numpy(session_df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values).transpose(0, 1).to(torch.float32)
        session_rir = torch.from_numpy(session_df['rir'].fillna(0).values).to(torch.float32)

        # Check length required for at least one inference call in mimic mode
        min_len_for_inference = winsize + (n_LSTM_windows - 1)

        if session_data.shape[1] >= min_len_for_inference:
            validation_session_data_label_pairs.append((session_data, session_rir))
        else:
            print(f"  Skipping session {session_id} for evaluation mimic - too short ({session_data.shape[1]} < {min_len_for_inference}) samples.")
    # --- END CHANGE ---

    if not validation_session_data_label_pairs: print("Error: No validation sessions long enough for mimic evaluation."); exit()
    print(f"Prepared {len(validation_session_data_label_pairs)} sessions for mimic evaluation.")

    # --- Create calibration loader ---
    num_calib_samples = min(len(val_dataset_full) // 2, 200)
    num_placeholder = len(val_dataset_full) - num_calib_samples
    if num_calib_samples > 0 and num_placeholder >= 0:
        # Ensure indices are generated if ConcatDataset doesn't support random_split directly
        indices = torch.randperm(len(val_dataset_full), generator=torch.Generator().manual_seed(42)).tolist()
        calib_indices = indices[:num_calib_samples]
        calib_dataset = Subset(val_dataset_full, calib_indices)
        # calib_dataset, _ = random_split(val_dataset_full, [num_calib_samples, num_placeholder], generator=torch.Generator().manual_seed(42))
    elif len(val_dataset_full) > 0: print("Warning: Using full validation set for calibration."); calib_dataset = val_dataset_full
    else: print("Error: No data for calibration loader."); exit()
    calib_batch_size = 32
    calib_loader = DataLoader(calib_dataset, batch_size=calib_batch_size, shuffle=False)
    print(f"Created calibration loader with ~{len(calib_loader)} batches.")

    # --- Evaluate Float Model (on ALL validation sessions) ---
    evaluate_model_mimic_live(model_fp32, validation_session_data_label_pairs, device, description="FP32 Model (All Val Sessions)") # Pass list of ALL tuples

    # --- Perform Quantization ---
    print("\nStarting quantization...")
    if len(calib_loader) == 0: print("Error: Calibration loader is empty."); model_quantized = None
    else: model_quantized = quantize_hybrid_static_dynamic(model_fp32, calib_loader, device)

    # --- Evaluate Quantized Model (on ALL validation sessions) ---
    if model_quantized:
        evaluate_model_mimic_live(model_quantized, validation_session_data_label_pairs, device, description="Quantized Model (All Val Sessions)") # Pass list of ALL tuples
    else: print("Quantization failed, skipping evaluation.")

    # --- Save Quantized Model (Optional) ---
    if model_quantized:
        quantized_model_save_path = 'quantized_hybrid_model.pth'
        print(f"\nSaving quantized model to: {quantized_model_save_path}")
        torch.save(model_quantized, quantized_model_save_path)

    print("\nScript finished.")