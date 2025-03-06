# https://randomnerdtutorials.com/raspberry-pi-analog-inputs-python-mcp3008/

# enable spi: sudo raspi-config nonint do_spi 0

from gpiozero import MCP3008
from time import sleep
import time

# refers to MCP3008 channel 1
hr = MCP3008(1)
start = time.time()
with open('heartbeat.csv', 'w') as f:
    f.write('time,heartrate\n')
    while True:
        val = hr.value * 3.3
        print(val)
        f.write(f'{(time.time() - start)*1000},{val}\n')
        sleep(0.01)
