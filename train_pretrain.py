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
from pathlib import Path

config = {
    'width': [64, 64, 128, 128, 256, 256, 512, 512, 1024],
    'depth': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'stem_out_c': 32,
    'stem_kernel': 7,
    'dropout': 0.1,
}
TRAIN=True

def moving_average(data, window_size=15):
    X_smooth = np.zeros(data.shape)
    for i,channel in enumerate(data):
        X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
    return torch.from_numpy(X_smooth).to(torch.float32)
def warp_time(data, max_scale=1.5):
    data = data.numpy()

    L = data.shape[1]
    scale = np.random.uniform(1, max_scale)
    new_L = int(L * scale)
    orig_t = np.linspace(0, 1, L)
    new_t = np.linspace(0, 1, new_L)

    warped = np.zeros((data.shape[0], new_L))
    for i, channel in enumerate(data):
        warped[i] = np.interp(new_t, orig_t, channel)
    
    # randomly crop to original length
    if new_L > L:
        start_idx = np.random.randint(0, new_L - L)
        warped = warped[:, start_idx:start_idx + L]

    return torch.from_numpy(warped).to(torch.float32)

def scale(data, low=0.6, high=1.4):
    return data*np.random.uniform(low, high)

class IMUDataset(Dataset):
    def __init__(self, df, winsize=250, stride=50, transform=None, aug=False):
        self.X = torch.from_numpy(df[['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values)
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
        X = self.X[:, start:end]
        X = moving_average(X)
        if self.aug:
            X = warp_time(X)
            X = scale(X)
        return X
    

HZ = 100
in_channels = 6

dfs = []
for file in Path('data/elshafei_data_cleaned').iterdir():
    df = pd.read_csv(file)
    df['session_id'] = file.stem.split('_')[0]
    dfs.append(df)
df_e = pd.concat(dfs, ignore_index=True)
session_ids_e = df_e['session_id'].unique()
train_ids_e, val_ids_e = train_test_split(session_ids_e, test_size=0.2, random_state=42)

df = pd.read_csv('data/data-strom.csv')
session_ids = df['session_id'].unique()
print(len(session_ids))
train_ids, val_ids = train_test_split(session_ids, test_size=0.2, random_state=42)

df = pd.concat([df, df_e], ignore_index=True)
train_ids = np.concatenate([train_ids, train_ids_e])
val_ids = np.concatenate([val_ids, val_ids_e])

df[['acc_x', 'acc_y', 'acc_z']] = (df[['acc_x', 'acc_y', 'acc_z']] / 2.0).clip(-1, 1)           # normalize accelerometer data from [-2g, 2g] to [-1, 1]
df[['gyr_x', 'gyr_y', 'gyr_z']] = (df[['gyr_x', 'gyr_y', 'gyr_z']] / 250.0).clip(-1, 1)         # normalize gyroscope data from [-250dps, 250dps] to [-1, 1]

# winsize_t = 5 # seconds
# stride_t = 0.01 # seconds
# winsize = int(winsize_t * HZ)
# stride = int(stride_t * HZ)

winsize = 1024
forecast_size = 256
config['forecast_size'] = forecast_size

# forecast_steps = 128
# winsize_ctxt = 1024
# winsize = winsize_ctxt + forecast_steps
stride = 2
print(winsize, stride)

train = df.loc[df['session_id'].isin(train_ids), ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']].values
norm = torch.from_numpy(train.mean(axis=0)), torch.from_numpy(train.std(axis=0))

# def transform(x):
#     return (x - norm[0]) / norm[1]
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

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.forecast_size = config['forecast_size']

        self.encoder = Encoder(config)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.conv_out_channels, self.forecast_size * in_channels),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        x = self.decoder(x)
        return x
    
model = AutoEncoder(config)
device = 'cuda'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# middle of window
forecast_slice = slice(winsize // 2 - forecast_size // 2, winsize // 2 + forecast_size // 2)
def get_masked(X):
    # return torch.cat([X[:,:,:forecast_slice.start], X[:,:,forecast_slice.stop:]], dim=-1)
    X = X.clone()
    X[:, :, forecast_slice] = 0
    return X

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

def train(model, epochs=10, outfile='best_model.pth'):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0.0
        for X in trainloader:
            X = X.to(device)
            optimizer.zero_grad()
            Xpred = model(get_masked(X)).view(X.size(0), in_channels, forecast_size)
            loss = criterion(Xpred, X[:,:,forecast_slice])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainloader)
        train_losses.append(train_loss)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X in valloader:
                X = X.to(device)
                Xpred = model(get_masked(X)).view(X.size(0), in_channels, forecast_size)
                loss = criterion(Xpred, X[:,:,forecast_slice])
                val_loss += loss.item()
            val_loss /= len(valloader)
            val_losses.append(val_loss)    

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save((model.state_dict(), model.config), outfile)
            print(f"*Epoch {epoch+1}/{epochs}, Loss: {train_loss:.07f}, Val Loss: {val_loss:.07f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.07f}, Val Loss: {val_loss:.07f}")

        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.savefig('loss.png')
        plt.legend()
        plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.show()
    
def evaluate(model, valloader, trainloader):
    train_loss = 0.0
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X in trainloader:
            X = X.to(device)
            Xpred = model(get_masked(X)).view(X.size(0), in_channels, forecast_size)
            loss = criterion(Xpred, X[:,:,forecast_slice])
            train_loss += loss.item()
        for X in valloader:
            X = X.to(device)
            Xpred = model(get_masked(X)).view(X.size(0), in_channels, forecast_size)
            loss = criterion(Xpred, X[:,:,forecast_slice])
            val_loss += loss.item()
        train_loss /= len(trainloader)
        val_loss /= len(valloader)
        print(f"Train Loss: {train_loss:.07f}\tValidation Loss: {val_loss:.07f}")

if TRAIN:
    train(model, epochs=100, outfile='best_model2.pth')

model.load_state_dict(torch.load('best_model2.pth')[0])
evaluate(model, valloader, trainloader)

model.load_state_dict(torch.load('best_model2.pth')[0])
model.eval()
X = valloader.dataset[200].unsqueeze(0)
X = X.to(device)
model.eval()
with torch.no_grad():
    Xpred = model(get_masked(X)).cpu().view(X.size(0), in_channels, forecast_size)

i=0
fig, axes = plt.subplots(2,1,figsize=(20, 10))
plt.subplot(2, 1, 1)
axes[0].plot(X[i,0].cpu(), label='Acc X')
axes[0].plot(X[i,1].cpu(), label='Acc Y')
axes[0].plot(X[i,2].cpu(), label='Acc Z')
axes[1].plot(X[i,3].cpu(), label='Gyr X')
axes[1].plot(X[i,4].cpu(), label='Gyr Y')
axes[1].plot(X[i,5].cpu(), label='Gyr Z')


axes[0].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,0], label='Pred Acc X', linestyle='--')
axes[0].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,1], label='Pred Acc Y', linestyle='--')
axes[0].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,2], label='Pred Acc Z', linestyle='--')
axes[1].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,3], label='Pred Gyr X', linestyle='--')
axes[1].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,4], label='Pred Gyr Y', linestyle='--')
axes[1].plot(np.arange(forecast_slice.start, forecast_slice.stop), Xpred[i,5], label='Pred Gyr Z', linestyle='--')

plt.legend(loc='upper left')
plt.savefig('forecast.png')