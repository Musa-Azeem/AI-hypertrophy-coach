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

import spidev
import smbus

WINDOW_TIME = 4     # seconds - for plotting
EMG_HZ = 2000       # In practice, this is about 1000 Hz - TODO: fix this
IMU_HZ = 200    
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

# Initialize SPI (for EMG)
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device (CS) 0
spi.max_speed_hz = 1350000  # 1.35 MHz
channel = 0

def sample_emg():
    # Read EMG Env - current units are (0,1023) - need to later convert to
    # 0-1 by dividing by 1023
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    value = ((adc[1] & 3) << 8) | adc[2] # 10-bit value from MCP3008 (0-1023)
    return value

ACCEL_YOUT_H = ACCEL_XOUT_H + 2
ACCEL_ZOUT_H = ACCEL_XOUT_H + 4
GYRO_YOUT_H = GYRO_XOUT_H + 2
GYRO_ZOUT_H = GYRO_XOUT_H + 4
def sample_imu():
    # read ACC - current units are (-32768, 32767) - need to later convert to 
    # g units (-2,+2) by dividing by 16384.0
    acc_x = read_imu_word(ACCEL_XOUT_H)
    acc_y = read_imu_word(ACCEL_YOUT_H)
    acc_z = read_imu_word(ACCEL_ZOUT_H)

    # read GYR - current units are (-32768, 32767) - need to layer convert to
    # deg/s: (-250, 250) by dividing by 131.0 
    gyr_x = read_imu_word(GYRO_XOUT_H)
    gyr_y = read_imu_word(GYRO_YOUT_H)
    gyr_z = read_imu_word(GYRO_ZOUT_H)
    return [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]

# Sample data scheduler
# def run_sampling_thread(data_dir, start_time, hz, write_hz):
#     def run():
#         sensor_data = []
#         scheduler = sched.scheduler(time.time, time.sleep)

#         def sample_data():
#             sensor_data.append((time.time() - start_time, sample_emg(), *sample_imu()))
#             scheduler.enter(1/hz, 1, sample_data)

#         def write_csv():
#             fn = 'data.csv'
#             columns = ['time', 'emg_env', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
#             df = pd.DataFrame(
#                 sensor_data, 
#                 columns=columns
#             )
#             if not (data_dir / fn).exists():
#                 df.to_csv(data_dir / fn, index=False)
#             else:
#                 df.to_csv(data_dir / fn, mode='a', header=False, index=False)
#             sensor_data[:df.shape[0]] = []
#             scheduler.enter(1/write_hz, 2, write_csv) # write to csv every 0.1s

#         scheduler.enter(1/hz, 1, sample_data)
#         scheduler.enter(1/write_hz, 2, write_csv)
#         scheduler.run()

#     threading.Thread(target=run, daemon=True).start()

# Sample data scheduler
def run_sampling_thread(data_dir, start_time, emg_hz, imu_hz, write_hz):
    emg_data = []
    imu_data = []

    def read_emg():
        emg_data.append((time.time(), sample_emg()))
    def read_imu():
        imu_data.append((time.time(), *sample_imu()))

    emg_fn = 'emg.csv'
    emg_columns = ['time', 'emg_env']
    def write_emg_csv():
        df = pd.DataFrame(
            emg_data, 
            columns=emg_columns
        )
        df['time'] -= start_time
        if not (data_dir / emg_fn).exists():
            df.to_csv(data_dir / emg_fn, index=False)
        else:
            df.to_csv(data_dir / emg_fn, mode='a', header=False, index=False)
        emg_data[:df.shape[0]] = []
    
    imu_fn = 'imu.csv'
    imu_columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    def write_imu_csv():
        df = pd.DataFrame(
            imu_data, 
            columns=imu_columns
        )
        df['time'] -= start_time
        if not (data_dir / imu_fn).exists():
            df.to_csv(data_dir / imu_fn, index=False)
        else:
            df.to_csv(data_dir / imu_fn, mode='a', header=False, index=False)
        imu_data[:df.shape[0]] = []

    emg_interval = 1.0 / emg_hz
    n_iter_write_emg = emg_hz / write_hz # number of iterations to wait before writing
    n_iter_imu = emg_hz / imu_hz
    def run_emg():
        i = 0
        while True:
            loop_start = time.perf_counter()

            read_emg()

            if i % n_iter_imu == 0:
                read_imu()
            if i == n_iter_write_emg:
                write_emg_csv()
                write_imu_csv()
                i = 0
            i += 1
            elapsed_time = time.perf_counter() - loop_start
            sleep_time = max(0, emg_interval - elapsed_time)
            time.sleep(sleep_time)

    imu_interval = 1.0 / imu_hz
    n_iter_write_imu = imu_hz / write_hz # number of iterations to wait before writing
    def run_imu():
        i = 0
        while True:
            loop_start = time.perf_counter()

            read_imu()

            if i == n_iter_write_imu:
                write_imu_csv()
                i = 0
            i += 1
            elapsed_time = time.perf_counter() - loop_start
            sleep_time = max(0, imu_interval - elapsed_time)
            time.sleep(sleep_time)

    threading.Thread(target=run_emg, daemon=True).start()
    # threading.Thread(target=run_imu, daemon=True).start()

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
    if not (data_dir / 'emg.csv').exists() or not (data_dir / 'imu.csv').exists():
        return no_update

    # TODO speed up this process
    # Read last 10 seconds of data
    filelines = subprocess.run(
        ['tail', '-n', str(WINDOW_TIME*EMG_HZ), data_dir / 'emg.csv'], 
        capture_output=True, 
        text=True
    ).stdout
    # TODO: don't hardcode t - this finds if header is present
    if filelines[0] == 't':
        emg = pd.read_csv(StringIO(filelines))
    else:
        emg = pd.read_csv(
            StringIO(filelines), 
            header=None, 
            names=['time', 'emg_env']
        )

    filelines = subprocess.run(
        ['tail', '-n', str(WINDOW_TIME*IMU_HZ), data_dir / 'imu.csv'], 
        capture_output=True, 
        text=True
    ).stdout
    # TODO: don't hardcode t - this finds if header is present
    if filelines[0] == 't':
        imu = pd.read_csv(StringIO(filelines))
    else:
        imu = pd.read_csv(
            StringIO(filelines), 
            header=None, 
            names=['time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        )

    fig = subplots.make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("EMG", "Accelerometer", "Gyroscope"),
    )
    fig.add_trace(go.Scatter(
        x=emg['time'], y=emg['emg_env'] / 1023 * 5.0, mode='lines', name='EMG'
    ), row=1, col=1)
    for col in ['acc_x', 'acc_y', 'acc_z']:
        fig.add_trace(go.Scatter(
            x=imu['time'], y=imu[col] / 16384.0, mode='lines', name=col
        ), row=2, col=1)
    for col in ['gyr_x', 'gyr_y', 'gyr_z']:
        fig.add_trace(go.Scatter(
            x=imu['time'], y=imu[col] / 131.0, mode='lines', name=col
        ), row=3, col=1)

    fig.update_layout(
        title_text="EMG and IMU Data",
        xaxis3_title="Time (s)",
        yaxis_title="EMG",
        yaxis2_title="deg/s",
        yaxis3_title="g m/s^2",
        xaxis = dict(
            range=[min(emg['time'].min(), imu['time'].min()), max(emg['time'].max(), imu['time'].max())],
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
        run_sampling_thread(directory, time.time(), EMG_HZ, IMU_HZ, WRITE_HZ)
        return str(directory)
    return no_update

if __name__ == "__main__":
    try:
        app.run_server(host='0.0.0.0')
    except KeyboardInterrupt:
        print("Stopping...")
        spi.close()