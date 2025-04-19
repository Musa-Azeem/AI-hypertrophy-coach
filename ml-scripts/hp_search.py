import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from scipy.stats import linregress
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import optuna
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import logging
import sys

STUDY_NAME = 'search256'
OUTDIR = Path(f'logs/{STUDY_NAME}')
DEVICE = 'cuda:0'
EPOCHS = 300

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
    
    label = torch.from_numpy(np.interp(new_t, orig_t, label)).to(torch.float32)
    # label = label * scale

    # randomly crop to original length
    if new_L > L:
        start_idx = np.random.randint(0, new_L - L)
        warped = warped[:, start_idx:start_idx + L]
        label = label[start_idx:start_idx + L]
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
    def __init__(self, df, winsize=250, stride=50, transform=None, aug=False):
        self.X = torch.from_numpy(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values)
        self.y = torch.from_numpy(df['rir'].values).to(torch.float32)
        if transform is not None:
            self.X = transform(self.X)
        self.X = self.X.transpose(0, 1).to(torch.float32)
        self.winsize = winsize
        self.stride = stride
        self.len = (self.X.shape[1] - winsize) // stride + 1
        self.aug = aug
    def __len__(self):
        return self.len
    def __getitem__(self, i):
        start = i * self.stride
        end = start + self.winsize
        y = self.y[start:end]#.mean()

        # FOR REGRESSION: y = y.mean() (single value)
        # y = y.mean()

        # FOR BINARY: 1 if y is last rep (< 1.5), 0 if not (single value)
        # y = torch.Tensor([1.0 if y.mean() < 1.5 else 0.0])

        # FOR SEGMENTATION: if y changes during window, y = 1 at that point. everywhere else is 0 (<winsize> values)
        y = segment_y(y)

        # FOR SEGMENTATION REGRESSION: y = time when y changes (single value)
        # diff = torch.diff(y)
        # y = torch.diff(y).argmin().unsqueeze(0) / self.winsize if diff.min() < 0 else torch.Tensor([-1])
        
        X = self.X[:, start:end]
        X = moving_average(X)
        if self.aug:
            X,y = warp_time(X, y)
            X = scale(X)
        return X, y
    

HZ = 100
in_channels = 6
df = pd.read_csv('data/data-strom.csv')
session_ids = df['session_id'].unique()
print(len(session_ids))
train_ids, val_ids = train_test_split(session_ids, test_size=0.2, random_state=42)

df[['acc_x', 'acc_y', 'acc_z']] = (df[['acc_x', 'acc_y', 'acc_z']] / 2.0).clip(-1, 1)           # normalize accelerometer data from [-2g, 2g] to [-1, 1]
df[['gyr_x', 'gyr_y', 'gyr_z']] = (df[['gyr_x', 'gyr_y', 'gyr_z']] / 250.0).clip(-1, 1)         # normalize gyroscope data from [-250dps, 250dps] to [-1, 1]

# winsize_t = 5 # seconds
# stride_t = 0.01 # seconds
# winsize = int(winsize_t * HZ)
# stride = int(stride_t * HZ)
winsize = 256
stride = 2
print(winsize, stride)

train = df.loc[df['session_id'].isin(train_ids), ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values
# norm = torch.from_numpy(train.mean(axis=0)), torch.from_numpy(train.std(axis=0))

# def transform(x):
    # return (x - norm[0]) / norm[1]
transform = None

train_dataset = ConcatDataset([IMUDataset(df[df['session_id'] == session_id], winsize, stride, transform, aug=True) for session_id in train_ids])
val_dataset = ConcatDataset([IMUDataset(df[df['session_id'] == session_id], winsize, stride, transform, aug=False) for session_id in val_ids])

len(train_dataset), len(val_dataset)

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
            nn.Conv1d(in_channels, self.stem_out_c, kernel_size=self.stem_kernel, padding=3),
            nn.BatchNorm1d(self.stem_out_c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
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
        x = self.encoder(x)
        # x = self.conv(x)
        x = self.ap(x).squeeze(-1)
        x = self.fc(x)
        # x = torch.repeat_interleave(x, 2, dim=1)
        return x
    def freeze(self, stop_idx=None):
        if stop_idx is None:
            stop_idx = len(self.encoder.encoder)
        for block in self.encoder.encoder[:stop_idx]:
            for param in block.parameters():
                param.requires_grad = False
        # for param in self.encoder.parameters():
            # param.requires_grad = False

def iou_metric(ypred, y_true):
    if y_true.ndim != 2:
        raise ValueError
    intersection = (ypred * y_true).sum(axis=1)
    union = (ypred + y_true - ypred * y_true).sum(axis=1)
    iou = (intersection / union)
    iou[iou.isnan()] = 0.0
    return iou.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.cat_criterion = nn.BCEWithLogitsLoss()
        self.reg_criterion = nn.MSELoss()
        # self.iou_criterion = iou_metric
        self.alpha = alpha
    def forward(self, ypred, y_true):
        probs = F.sigmoid(ypred)
        cat_loss = self.cat_criterion(ypred, y_true)
        reg_loss = self.reg_criterion(probs, y_true)
        return self.alpha*cat_loss + (1-self.alpha)*reg_loss

def objective(trial):
    outdir = OUTDIR / f'{trial.number}'
    writer = SummaryWriter(outdir)

    n_stages = trial.suggest_int("n_stages", 3, 6)
    config = dict(
        stem_out_c = 2**trial.suggest_int("stem_out_c", 6, 10),
        depth = [trial.suggest_int(f"depth_{i}", 1, 2) for i in range(n_stages)],
        width = [2**trial.suggest_int(f"width_{i}", 6, 10) for i in range(n_stages-1)],
        stem_kernel = trial.suggest_int("stem_kernel", 3, 7, step=2),
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        dropout = trial.suggest_float("dropout", 0.0, 0.25),
        batch_size = 128,#2**trial.suggest_int("batch_size", 5, 8),
        device = DEVICE,
        train_winsize = winsize,
        train_stride = stride
    )
    config['width'].append(2**trial.suggest_int("width_last", 7, 10)) # last width determines linear layer input size

    device = config['device']

    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = ConvNet(config).to(device)
    criterion = SegmentationLoss(0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    print(f'Trial: {trial.number} - {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters - k {config["stem_kernel"]} - stem: {config["stem_out_c"]} - depth: {config["depth"]} - width: {config["width"]} - lr: {config["learning_rate"]:.4f} - weight_decay: {config["weight_decay"]:.4f} - dropout: {config["dropout"]:.2f} - batch_size: {config["batch_size"]}')

    best_f1_epoch = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    patience = 15
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
        val_ioui = []
        ypreds = []
        ys = []
        with torch.no_grad():
            for X, y in valloader:
                X, y = X.to(device), y.to(device)
                ypred = model(X)
                loss = criterion(ypred, y)
                val_lossi.append(loss.item())
                ypred = F.sigmoid(ypred).round()
                val_ioui.append(iou_metric(ypred, y).item())
                ypreds.append(ypred.cpu())
                ys.append(y.cpu())
        val_loss = np.mean(val_lossi)
        val_iou = np.mean(val_ioui)
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
        writer.add_scalar('val/iou', val_iou, epoch)

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

sampler = optuna.samplers.TPESampler(n_startup_trials=10, prior_weight=0.9)
study = optuna.create_study(
    study_name=STUDY_NAME,
    sampler=sampler,
    direction="maximize", 
    storage=f"sqlite:///{STUDY_NAME}.db", 
    load_if_exists=True
)

study.optimize(objective, n_trials=1000)