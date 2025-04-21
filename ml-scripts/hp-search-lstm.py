import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from torch import nn
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import optuna
import logging
import sys

STUDY_NAME = 'search-lstm'
OUTDIR = Path(f'logs/{STUDY_NAME}')
EPOCHS = 50
DEVICE = 'cuda'

def moving_average(data, window_size=15):
    X_smooth = np.zeros(data.shape)
    for i,channel in enumerate(data):
        X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
    return torch.from_numpy(X_smooth).to(torch.float32)
def warp_time(data, label, max_scale=1.5):
    data = data.numpy()

    L = data.shape[1]
    scale = np.random.uniform(1, max_scale)
    new_L = int(L * scale)
    orig_t = np.linspace(0, 1, L)
    new_t = np.linspace(0, 1, new_L)

    warped = np.zeros((data.shape[0], new_L))
    for i, channel in enumerate(data):
        warped[i] = np.interp(new_t, orig_t, channel)
    
    # label = torch.from_numpy(np.interp(new_t, orig_t, label)).to(torch.float32)
    # label = label * scale

    # randomly crop to original length
    if new_L > L:
        start_idx = np.random.randint(0, new_L - L)
        warped = warped[:, start_idx:start_idx + L]
        # label = label[start_idx:start_idx + L]
        # label = label - (start_idx / L)

    return torch.from_numpy(warped).to(torch.float32), label

def scale(data, low=0.6, high=1.4):
    return data*np.random.uniform(low, high)

def segment_y(y, t=40):
    end_rep_markers = torch.where(torch.diff(y) < 0)[0]
    y = torch.zeros_like(y)
    starts = (end_rep_markers - t).clamp(0)
    ends = (end_rep_markers + t).clamp(0, y.shape[0])
    for start,end in zip(starts, ends):
        y[start: end] = 1
    return y
class IMUDataset(Dataset):
    def __init__(self, df, winsize, stride, full_winsize, n_LSTM_windows, LSTM_stride, transform=None, aug=False):
        self.X = torch.from_numpy(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values)
        self.y = torch.from_numpy(df['rir'].values).to(torch.float32)
        if transform is not None:
            self.X = transform(self.X)
        self.X = self.X.transpose(0, 1).to(torch.float32)
        self.winsize = winsize
        self.stride = stride
        self.full_winsize = full_winsize
        self.n_LSTM_windows = n_LSTM_windows
        self.LSTM_stride = LSTM_stride
        self.len = (self.X.shape[1] - full_winsize) // stride + 1
        self.aug = aug
    def __len__(self):
        return self.len
    def __getitem__(self, i):
        if i >= self.len:
            raise IndexError
        start = i * self.stride
        end = start + self.full_winsize

        X = self.X[:, start:end]
        y = self.y[start:end]

        X = moving_average(X)
        if self.aug:
            X,y = warp_time(X,y)
            X = scale(X)
        
        # window by LSTM stride
        X = X.unfold(1, self.winsize, self.LSTM_stride).permute(1,0,2)
        y = y.unfold(0, self.winsize, self.LSTM_stride)

        # FOR REGRESSION: y = y.mean() (single value)
        # y = y.mean()

        # FOR BINARY: 1 if y is last rep (< 1.5), 0 if not (single value)
        # y = torch.Tensor([1.0 if y.mean() < 1.5 else 0.0])
        y = (y.mean(dim=1) < 0.5).float().unsqueeze(1)

        # FOR SEGMENTATION: if y changes during window, y = 1 at that point. everywhere else is 0 (<winsize> values)
        # y = segment_y(y)

        # FOR SEGMENTATION REGRESSION: y = time when y changes (single value)
        # diff = torch.diff(y)
        # y = torch.diff(y).argmin().unsqueeze(0) / self.winsize if diff.min() < 0 else torch.Tensor([-1])
        
        return X, y
HZ = 100
in_channels = 6
df = pd.read_csv('data/data.csv')
session_ids = df['session_id'].unique()
print(len(session_ids))
train_ids, val_ids = train_test_split(session_ids, test_size=0.2, random_state=42)
train_ids = np.array([train_id for train_id in train_ids if df[df.session_id == train_id].failure.sum()])
val_ids = np.array([val_id for val_id in val_ids if df[df.session_id == val_id].failure.sum()])

df[['acc_x', 'acc_y', 'acc_z']] = (df[['acc_x', 'acc_y', 'acc_z']] / 2.0).clip(-1, 1)           # normalize accelerometer data from [-2g, 2g] to [-1, 1]
df[['gyr_x', 'gyr_y', 'gyr_z']] = (df[['gyr_x', 'gyr_y', 'gyr_z']] / 250.0).clip(-1, 1)         # normalize gyroscope data from [-250dps, 250dps] to [-1, 1]

# winsize_t = 5 # seconds
# stride_t = 0.01 # seconds
# winsize = int(winsize_t * HZ)
# stride = int(stride_t * HZ)
winsize = 256
stride = 2

n_LSTM_windows = 16
LSTM_stride = 64
print(winsize, stride)

full_winsize = LSTM_stride * (n_LSTM_windows-1) + winsize
print(full_winsize)

train = df.loc[df['session_id'].isin(train_ids), ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values
# norm = torch.from_numpy(train.mean(axis=0)), torch.from_numpy(train.std(axis=0))

# def transform(x):
    # return (x - norm[0]) / norm[1]
transform = None

train_dataset = ConcatDataset([IMUDataset(
    df[df['session_id'] == session_id], 
    winsize, stride,
    full_winsize, n_LSTM_windows, LSTM_stride,
    transform, aug=True
) for session_id in train_ids])
val_dataset = ConcatDataset([IMUDataset(
    df[df['session_id'] == session_id], 
    winsize, stride, 
    full_winsize, n_LSTM_windows, LSTM_stride,
    transform, aug=False
) for session_id in val_ids])

len(train_dataset), len(val_dataset)

def get_time_in_reps(preds, stride, winsize):
    # preds: N x T x C
    time_in_rep = []
    # consolidated = []
    end_reps = []
    for pred in preds: # each element in batch
        consolidatedi = []
        for i,predi in enumerate(pred): # each element in time
            upper = stride if i < len(pred) - 1 else len(predi)
            consolidatedi.append(predi[:upper])
        consolidatedi = torch.cat(consolidatedi, axis=0)
        consolidatedi = torch.from_numpy(np.convolve(consolidatedi, np.ones(100)/100, mode='same').round()).to(torch.float32)

        time_in_repi = torch.zeros_like(consolidatedi).to(torch.float32)
        end_repsi = []
        if consolidatedi.sum() > 0:
            diff = np.diff(consolidatedi)
            starts = np.where(diff > 0)[0]
            ends = np.where(diff < 0)[0]
            if len(starts) == 0 or len(ends) == 0:
                if len(starts) == 0:
                    # len(end) == 1 -> start is the beginning of session
                    starts = np.array([0])
                elif len(ends) == 0:
                    # len(start) == 1 -> end is the end of session
                    ends = np.array([len(pred[0])])
            else:
                if starts[0] > ends[0]:
                    # first end has no start -> first start is the beginning of session
                    starts = np.concatenate([[0], starts])
                if ends[-1] < starts[-1]:
                    # last start has no end -> last end is the end of session
                    ends = np.concatenate([ends, [len(pred[0])]])
            for start,end in zip(starts, ends):
                end_repsi.append((start + end) // 2)
            
            for i in range(1, len(end_repsi)):
                time_in_repi[end_repsi[i-1]:end_repsi[i]] = (end_repsi[i] - end_repsi[i-1]) / full_winsize
                # time_in_rep.append(end_repsi[i] - end_repsi[i-1])

        end_reps.append(end_repsi)
        # consolidated.append(consolidatedi)
        time_in_rep.append(time_in_repi)
    # preds = torch.stack(consolidated, axis=0)
    time_in_rep = torch.stack(time_in_rep, axis=0)

    # rewindow
    time_in_rep = time_in_rep.unfold(1, winsize, stride)

    return time_in_rep

class ResBlock(nn.Module):
    # One layer of convolutional block with batchnorm, relu and dropout
    def __init__(
            self, in_channels, out_channels,
            kernel_size=3, stride=1, dropout=0.0,
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.skip = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, stride=stride
        ) if in_channels != out_channels or stride > 1 else nn.Identity()
    def forward(self, x):
        return self.block(x) + self.skip(x)
    
class DepthBlock(nn.Module):
    # "depth" number of ConvBlocks with downsample on the first block
    def __init__(
            self, depth, in_channels, out_channels,
            kernel_size=3, downsample_stride=2, 
            dropout=0.0
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResBlock(
                in_channels=in_channels if i == 0 else out_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size, 
                stride=downsample_stride if i == 0 else 1,
                dropout=dropout
            )
            for i in range(depth)
        ])
    def forward(self, x):
        return self.blocks(x)
 
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.width = config['width']
        self.depth = config['depth']
        self.stem_out_c = config['stem_out_c']
        self.stem_kernel = config['stem_kernel']
        self.dropout = config['dropout']

        if len(self.width) != len(self.depth):
            raise ValueError('Width and depth must have the same length')
        self.conv_out_channels = self.stem_out_c if len(self.width) == 0 else self.width[-1]

        self.encoder = nn.Sequential(
            # nn.Conv1d(in_channels, self.stem_out_c, kernel_size=self.stem_kernel, padding=self.stem_kernel // 2, stride=2),
            # nn.BatchNorm1d(self.stem_out_c),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            ResBlock(in_channels, self.stem_out_c, kernel_size=self.stem_kernel, stride=2),
            *[DepthBlock(
                depth=self.depth[i],
                in_channels=self.stem_out_c if i == 0 else self.width[i-1], 
                out_channels=self.width[i],
                dropout=self.dropout, 
            ) for i in range(len(self.width))]
        )
    def forward(self, x):
        return self.encoder(x)

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.ap = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(512, winsize // 2)
        self.fc = nn.Linear(self.encoder.conv_out_channels, winsize)
    def forward(self, x):
        emb = self.encoder(x)
        # x = self.conv(x)
        x = self.ap(emb).squeeze(-1)
        seg_logits = self.fc(x)
        # x = torch.repeat_interleave(x, 2, dim=1)
        return emb, seg_logits
    
    def freeze(self, stop_idx=None):
        if stop_idx is None:
            stop_idx = len(self.encoder.encoder)
        for block in self.encoder.encoder[:stop_idx]:
            for param in block.parameters():
                param.requires_grad = False

class LSTMNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = 6


        self.encoder = ConvNet(config)
        hidden_size = config['lstm_config']['hidden_size']
        # encoder_proj_channels = config['lstm_config']['encoder_proj_channels']
        skip_channels = config['lstm_config']['skip_channels']
        num_layers = config['lstm_config']['num_layers']
        dropout = config['lstm_config']['dropout']
        linear_hidden_size = config['lstm_config']['linear_hidden_size']

        # # Project encoder output to hidden size
        # self.conv_proj = nn.Conv1d(
        #     self.encoder.encoder.conv_out_channels, 
        #     encoder_proj_channels, 
        #     kernel_size=1
        # )

        # Convolution layer from input signal to skip_channels size
        self.conv_skip = nn.Sequential(
            nn.Conv1d(in_channels, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(skip_channels),
            nn.ReLU(),
            # ResBlock(in_channels, skip_channels, kernel_size=7, stride=1),
        )
        self.ap = nn.AdaptiveAvgPool1d(1)

        # Project encoder, skip layer, and time_in_reps to lstm input size
        self.lstm_proj = nn.Linear(
            # encoder_proj_channels + skip_channels + winsize,
            self.encoder.encoder.conv_out_channels + skip_channels + winsize,
            hidden_size
        )

        # LSTM layer on projected encoder, skip layer, and time_in_reps
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )

        # Linear layer to predict the output
        if linear_hidden_size == 0:
            self.fc = nn.Linear(hidden_size, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, linear_hidden_size),
                nn.ReLU(),
                nn.Linear(linear_hidden_size, 1)
            )
    def forward(self, x):
        N, T, C, L = x.shape

        x = x.view(N*T, C, L)
        
        # Run encoder to get embeddings and segmentation logits
        x_seg, seg_logits = self.encoder(x)

        # x_seg = self.conv_proj(x_seg)
        x_seg = self.ap(x_seg).squeeze(-1)
        x_seg = x_seg.view(N, T, -1)

        # Project segmentation logits to seg_proj_size
        time_in_rep = get_time_in_reps(seg_logits.view(N, T, L).detach().cpu(), LSTM_stride, winsize).to(x.device)

        x_skip = self.conv_skip(x)
        x_skip = self.ap(x_skip).squeeze(-1)
        x_skip = x_skip.view(N, T, -1)

        x = torch.cat([x_seg, x_skip, time_in_rep], dim=2)
        x = self.lstm_proj(x)

        o, (h,c) = self.lstm(x)
        # x = self.fc(o[:, -1, :]) # predict for last time step
        x = self.fc(o)        # predict for all time steps
        return x
    
    def get_optimizer(self, lr, weight_decay=1e-4, betas=(0.9, 0.999)):
        # AdamW optimzer - apply weight decay to linear and conv weights
        # but not to biases and batchnorm layers
        params = self.named_parameters()
        decay_params = [p for n,p in params if p.dim() >= 2]
        no_decay_params = [p for n,p in params if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], betas=betas, lr=lr)
        return optimizer

class Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([20]).to(device))
        # self.loss = nn.MSELoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
        # return self.loss(y_pred, y_true[:,-1])

def objective(trial):
    outdir = OUTDIR / f'{trial.number}'
    writer = SummaryWriter(outdir)

    config = json.load(open('logs/search256-3/56/config.json'))
    weights = torch.load('logs/search256-3/56/best_model.pth')

    lstm_config = {
        'num_layers': trial.suggest_int('num_layers', 1, 8),
        # 'encoder_proj_channels': 2**trial.suggest_int('encoder_proj_channels', 5, 10),
        'skip_channels': 2**trial.suggest_int('skip_channels', 5, 10),
        'hidden_size': 2**trial.suggest_int('hidden_size', 5, 10),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
        'linear_hidden_size': 0,
        'freeze': True,
        'num_windows': n_LSTM_windows,
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
    }

    config['lstm_config'] = lstm_config

    # weights, config = torch.load('../best_model-class-83_2.pth')
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    device = DEVICE

    model = LSTMNet(config).to(device)
    model.encoder.load_state_dict(weights)
    model.encoder.freeze()

    criterion = Loss(device)#nn.BCEWithLogitsLoss()
    optimizer = model.get_optimizer(lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print(f'Trial: {trial.number} - {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters - num_layers: {lstm_config["num_layers"]} - skip_channels: {lstm_config["skip_channels"]} - hidden_size: {lstm_config["hidden_size"]}')

    best_f1_epoch = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    patience = 10
    early_stop = 0
    min_delta = 0.001

    pbar = tqdm(range(EPOCHS))
    for epoch in pbar:
        model.train()
        train_lossi = []
        for X,y in trainloader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_lossi.append(loss.item())
        train_loss = np.mean(train_lossi)

        model.eval()
        val_lossi = []
        ypreds = []
        ys = []
        with torch.no_grad():
            for X, y in valloader:
                X, y = X.to(device), y.to(device)
                ypred = model(X)
                loss = criterion(ypred, y)
                val_lossi.append(loss.item())
                ypred = F.sigmoid(ypred).round()
                ypreds.append(ypred.cpu())
                ys.append(y.cpu())
        val_loss = np.mean(val_lossi)
        ypreds = torch.cat(ypreds).numpy().flatten()
        ys = torch.cat(ys).numpy().flatten()

        val_acc = (ypreds == ys).mean()
        precision, recall, val_f1, _ = precision_recall_fscore_support(ys, ypreds, average='binary')

        # Early stopping
        if val_f1 <= best_val_f1 or val_f1 - best_val_f1 < min_delta:        
            early_stop += 1
        else:
            early_stop = 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_f1 > best_val_f1:
            best_f1_epoch = epoch
            best_val_f1 = val_f1
            torch.save(model.state_dict(), outdir / 'best_model.pth')
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('val/f1', val_f1, epoch)
        writer.add_scalar('val/precision', precision, epoch)
        writer.add_scalar('val/recall', recall, epoch)

        trial.report(val_f1, epoch)
        if early_stop >= patience:
            break

        pbar.set_description(f'{val_f1:.2f}')

    config['best_val_loss'] = best_val_loss
    config['best_f1_epoch'] = best_f1_epoch
    config['best_val_f1'] = best_val_f1
    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    writer.close()

    return best_val_f1

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

sampler = optuna.samplers.TPESampler(n_startup_trials=5, prior_weight=0.9)
study = optuna.create_study(
    study_name=STUDY_NAME,
    sampler=sampler,
    direction="maximize", 
    storage=f"sqlite:///{STUDY_NAME}.db", 
    load_if_exists=True
)

study.optimize(objective, n_trials=1000)