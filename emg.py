# https://randomnerdtutorials.com/raspberry-pi-analog-inputs-python-mcp3008/

# enable spi: sudo raspi-config nonint do_spi 0

from gpiozero import MCP3008
from time import sleep
import time

# refers to MCP3008 channel 0
emg = MCP3008(0)

with open('recordings2/curl.csv', 'w') as f:
    f.write(f'time,emg')
    while True:
        val = emg.value * 3.3
        f.write(f'{time.time()},{val}\n')
        sleep(1/100)
