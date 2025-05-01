import time
import threading
import pandas as pd
import dash
from dash import dcc, html, Output, Input, no_update, State
import plotly.express as px
from plotly import subplots
import plotly.graph_objects as go
import sched
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
from io import StringIO
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from gpiozero import MCP3008
import smbus

WINDOW_TIME = 4     # seconds - for plotting
HZ = 250            # In practice, this is about 100 Hz - TODO: fix this
in_channels = 6
WRITE_HZ = 2        # write to csv every 0.5s

# MPU6050 I2C address
MPU6050_ADDR = 0x68

# Register addresses
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

bus = smbus.SMBus(1)    # initialize I2C bus
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)    # wake up MPU6050
def read_imu_word(adr):
    high = bus.read_byte_data(MPU6050_ADDR, adr)
    low = bus.read_byte_data(MPU6050_ADDR, adr + 1)
    value = (high << 8) + low
    if value >= 0x8000:  # MSB is high - convert to signed
        value = -((65535 - value) + 1)
    return value

emg = MCP3008(0)

# Simulated data sampling functions
def sample_emg():
    return emg.value * 3.3 # read value from ADC * ref voltage (3.3) to get original value

def sample_imu():
    # read ACC - convert to g units: (-2,+2) from (-32768, 32767)
    # TODO move these operations to somewhere else to speed up sampling
    acc_x = read_imu_word(ACCEL_XOUT_H) / 16384.0
    acc_y = read_imu_word(ACCEL_XOUT_H + 2) / 16384.0
    acc_z = read_imu_word(ACCEL_XOUT_H + 4) / 16384.0

    # read GYR - convert to deg/s: (-250, 250) from (-32768, 32767)
    gyr_x = read_imu_word(GYRO_XOUT_H) / 131.0
    gyr_y = read_imu_word(GYRO_XOUT_H + 2) / 131.0
    gyr_z = read_imu_word(GYRO_XOUT_H + 4) / 131.0
    return [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]

def moving_average(data, window_size=15):
    X_smooth = np.zeros(data.shape)
    for i,channel in enumerate(data):
        X_smooth[i] = np.convolve(channel, np.ones(window_size)/window_size, mode='same')
    return torch.from_numpy(X_smooth).to(torch.float32)

winsize = 256
n_LSTM_windows = 32
LSTM_stride = 64
full_winsize = LSTM_stride * (n_LSTM_windows-1) + winsize

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
        for param in self.fc.parameters():
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
        seg_logits = F.sigmoid(seg_logits).round()
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
    def __init__(self, device, pos_weight):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))
        # self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.MSELoss()
    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
        # return self.loss(y_pred, y_true[:,-1])

# Sample data scheduler
def run_sampling_thread(data_dir, start_time, hz, write_hz):
    def run():
        sensor_data = []
        input_windows = []
        window = []

        scheduler = sched.scheduler(time.time, time.sleep)

        weights, config = torch.load('../best_model-3binary-77.pth', map_location=torch.device("cpu"))
        model = LSTMNet(config)
        model.load_state_dict(weights)
        model.eval()
        ypreds = []

        def sample_data():
            sensor_data.append((time.time() - start_time, sample_emg(), *sample_imu()))

            if len(window) != 256:
                window.append(sample_imu())
            else:
                if len(input_windows) == 32:
                    del input_windows[0]
                input_windows.append(window)

                input_windows_copy = torch.tensor(input_windows).transpose(1, 2).to(torch.float32).unsqueeze(0)

                with torch.no_grad():
                    conf = model(input_windows_copy)
                    conf = F.sigmoid(conf[:,-1, 0])
                    ypreds.append(conf.round())
                    print(torch.tensor(ypreds).flatten())

                del window[:64]
                window.append(sample_imu())

            scheduler.enter(1/hz, 1, sample_data)

        def write_csv():
            fn = 'data.csv'
            columns = ['time', 'emg_env', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
            df = pd.DataFrame(
                sensor_data, 
                columns=columns
            )
            if not (data_dir / fn).exists():
                df.to_csv(data_dir / fn, index=False)
            else:
                df.to_csv(data_dir / fn, mode='a', header=False, index=False)

            sensor_data[:df.shape[0]] = []
            scheduler.enter(1/write_hz, 2, write_csv) # write to csv every 0.1s

        scheduler.enter(1/hz, 1, sample_data)
        scheduler.enter(1/write_hz, 2, write_csv)
        scheduler.run()

    threading.Thread(target=run, daemon=True).start()
    

def run_scheduler_thread(schedular):
    def run():
        schedular.run()


app = dash.Dash(__name__)

last_info = json.load(open('last_info.json', 'r')) if Path('last_info.json').exists() else {}

# Dash Layout
app.layout = html.Div([
    html.H1("AHC Data Collection Dashboard"),
    dcc.Graph(id="data-plot", style={"width": "80%", "height": "80vh", "padding": "10px"}),
    html.Div([
        html.Button("Report End of Rep", id="record-btn", n_clicks=0, style={"marginBottom": "10px"}),
        html.Div([
            html.Label("Name:"),
            dcc.Input(id="name-input", type="text", placeholder="Enter name", value=last_info.get("name")),
        ], className="input-group"),
        html.Div([
            html.Label("Age:"),
            dcc.Input(id="age-input", type="number", placeholder="Enter age", value=last_info.get("age")),
        ], className="input-group"),
        html.Div([
            html.Label("Weight (lb):"),
            dcc.Input(id="weight-input", type="number", placeholder="Enter weight", value=last_info.get("weight")),
        ], className="input-group"),
        html.Div([
            html.Label("Sex:"),
            dcc.Input(id="sex-input", type="text", placeholder="Enter sex", value=last_info.get("sex")),
        ], className="input-group"),
        html.Div([
            html.Label("Location:"),
            dcc.Input(id="location-input", type="text", placeholder="Enter location", value=last_info.get("location")),
        ], className="input-group"),
        html.Div([
            html.Label("Machine Weight (lb)"),
            dcc.Input(id="machine-weight-input", type="number", placeholder="Enter machine weight", value=last_info.get("machine_weight")),
        ], className="input-group"),
        html.Button("Start Session", id="start-session-btn", n_clicks=0, style={"marginTop": "15px"}),
    ], className="form-container"),
    dcc.Interval(id="update-interval", interval=500, n_intervals=0),
    dcc.Store(id="data_dir", data=None),
    dcc.Store(id="end_rep_markers", data=[]),
], style={
    "width": "95vw", 
    "height": "95vh",
    "textAlign": "center",
    "display": "flex", 
    "flexDirection": "column", 
    "alignItems": "center", 
    "justifyContent": "center",
})

@app.callback(
    Output("data-plot", "figure"),
    Input("update-interval", "n_intervals"),
    Input("data_dir", "data"),
    prevent_initial_call=True
)
def update_plots(n_intervals, data_dir):
    if data_dir is None:
        return no_update
    data_dir = Path(data_dir)
    if not (data_dir / 'data.csv').exists():
        return no_update

    # TODO speed up this process
    # Read last 10 seconds of data
    filelines = subprocess.run(
        ['tail', '-n', str(WINDOW_TIME*HZ), data_dir / 'data.csv'], 
        capture_output=True, 
        text=True
    ).stdout
    # TODO: don't hardcode t - this finds if header is present
    if filelines[0] == 't':
        df = pd.read_csv(StringIO(filelines))
    else:
        df = pd.read_csv(
            StringIO(filelines), 
            header=None, 
            names=['time', 'emg_env', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        )

    fig = subplots.make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("EMG", "Accelerometer", "Gyroscope"),
    )
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['emg_env'], mode='lines', name='EMG'
    ), row=1, col=1)
    for col in ['acc_x', 'acc_y', 'acc_z']:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[col], mode='lines', name=col
        ), row=2, col=1)
    for col in ['gyr_x', 'gyr_y', 'gyr_z']:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df[col], mode='lines', name=col
        ), row=3, col=1)

    fig.update_layout(
        title_text="EMG and IMU Data",
        xaxis3_title="Time (s)",
        yaxis_title="EMG",
        yaxis2_title="deg/s",
        yaxis3_title="g m/s^2",
        xaxis = dict(
            range=[df['time'].min(), df['time'].min() + WINDOW_TIME],
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

@app.callback(
    Output("end_rep_markers", "data"),
    Input("end_rep_markers", "data"),
    Input("data_dir", "data"),
    Input("record-btn", "n_clicks")
)
def record_timestamp(end_rep_markers, data_dir, n_clicks):
    if n_clicks > len(end_rep_markers):
        t = time.time()
        info = json.load(open(Path(data_dir) / 'info.json'))
        end_rep_markers = end_rep_markers + [t - info['start_time']]
        info['end_rep_markers'] = end_rep_markers
        json.dump(info, open(Path(data_dir) / 'info.json', 'w'), indent=4)
    return end_rep_markers

@app.callback(
    Output('data_dir', 'data'),
    Input("start-session-btn", "n_clicks"),
    Input('data_dir', 'data'),
    State("name-input", "value"),
    State("age-input", "value"),
    State("weight-input", "value"),
    State("sex-input", "value"),
    State("location-input", "value"),
    State("machine-weight-input", "value"),
    prevent_initial_call=True
)
def start_session(n_clicks, data_dir, name, age, weight, sex, location, machine_weight):
    if n_clicks > 0 and data_dir is None:
        if not name or not age or not weight:
            return no_update

        directory = Path(f'recordings/{name}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
        directory.mkdir(exist_ok=True, parents=True)
        info = dict(
            name=name,
            age=age,
            weight=weight,
            sex=sex,
            location=location,
            machine_weight=machine_weight,
            start_time=time.time(),
            notes="",
        )
        json.dump(info, open(directory / 'info.json', 'w'), indent=4)
        json.dump(info, open('last_info.json', 'w'), indent=4)
        run_sampling_thread(directory, time.time(), HZ, WRITE_HZ)
        return str(directory)
    return no_update

if __name__ == "__main__":
    app.run_server(host='0.0.0.0')