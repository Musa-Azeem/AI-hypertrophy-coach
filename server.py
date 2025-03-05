import time
import threading
import pandas as pd
import dash
from dash import dcc, html, Output, Input, no_update, State
import plotly.express as px
from plotly import subplots
import plotly.graph_objects as go
import sched

app_start_time = time.time()

# Simulated data sampling functions
def sample_emg():
    from random import random
    return random()  # Simulating EMG signal

def sample_imu():
    from random import random
    return [random() for _ in range(6)]  # Simulating IMU 6-axis data

app = dash.Dash(__name__)

# Data storage
emg_data = []
imu_data = []
timestamps = []
button_presses = []
session_info = {}
session_active = False

scheduler = sched.scheduler(time.time, time.sleep)

def sample_data():
    emg_data.append((time.time(), sample_emg()))
    scheduler.enter(1/500, 1, sample_data)

def sample_imu_data():
    imu_data.append((time.time(), *sample_imu()))
    scheduler.enter(1/100, 1, sample_imu_data)

scheduler.enter(1/500, 1, sample_data)
scheduler.enter(1/100, 1, sample_imu_data)

# Dash Layout
app.layout = html.Div([
    html.H1("AHC Data Collection Dashboard"),
    dcc.Graph(id="data-plot"),
    html.Button("Record Timestamp", id="record-btn", n_clicks=0),
    html.Div([
        html.Label("Name:"),
        dcc.Input(id="name-input", type="text", placeholder="Enter name"),
        html.Label("Age:"),
        dcc.Input(id="age-input", type="number", placeholder="Enter age"),
        html.Label("Weight (kg):"),
        dcc.Input(id="weight-input", type="number", placeholder="Enter weight"),
        html.Button("Start Session", id="start-session-btn", n_clicks=0),
    ]),
    dcc.Interval(id="update-interval", interval=100, n_intervals=0) # callback every 100ms
])

# Callbacks to update plots and record button presses
@app.callback(
    # Output("emg-plot", "figure"),
    # Output("imu-plot", "figure"),

    Output("data-plot", "figure"),
    Input("update-interval", "n_intervals")
)
def update_plots(_):
    time_window = 10 # seconds

    # collect and plot data for the last 10 seconds for both EMG and IMU
    df_emg = pd.DataFrame(emg_data[-time_window*500:], columns=["Time", "EMG"])
    df_emg['Time'] = df_emg['Time'] - app_start_time

    # Convert IMU data to a DataFrame
    df_imu = pd.DataFrame(imu_data[-time_window*100:], columns=["Time", "X", "Y", "Z", "Roll", "Pitch", "Yaw"])
    df_imu['Time'] = df_imu['Time'] - app_start_time

    fig = subplots.make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("EMG", "IMU"),
    )
    fig.add_trace(go.Scatter(
        x=df_emg['Time'], y=df_emg['EMG'], mode='lines', name='EMG'
    ), row=1, col=1)
    for col in ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]:
        fig.add_trace(go.Scatter(
            x=df_imu['Time'], y=df_imu[col], mode='lines', name=col
        ), row=2, col=1)

    fig.update_layout(
        height=800, 
        width=1500, 
        title_text="EMG and IMU Data",
        xaxis_title="Time (s)",
        yaxis_title="EMG",
        yaxis2_title="IMU",
        xaxis = dict(
            range=[df_emg['Time'].min(), df_emg['Time'].max() + 1]
        )
    )
    return fig

@app.callback(
    Output("record-btn", "n_clicks"),  # Dummy Output
    Input("record-btn", "n_clicks")
)
def record_timestamp(n_clicks):
    if n_clicks > len(button_presses):
        button_presses.append(time.time())
    print('button')
    return no_update

@app.callback(
    Output("start-session-btn", "n_clicks"),
    Input("start-session-btn", "n_clicks"),
    State("name-input", "value"),
    State("age-input", "value"),
    State("weight-input", "value")
)
def start_session(n_clicks, name, age, weight):
    global session_active
    if n_clicks > 0 and not session_active:
        session_active = True
        session_info["name"] = name
        session_info["age"] = age
        session_info["weight"] = weight
        threading.Thread(target=scheduler.run, daemon=True).start()
    return no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
