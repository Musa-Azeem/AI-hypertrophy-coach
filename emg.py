# https://randomnerdtutorials.com/raspberry-pi-analog-inputs-python-mcp3008/

# enable spi: sudo raspi-config nonint do_spi 0

from gpiozero import MCP3008
from time import sleep

# refers to MCP3008 channel 0
emg = MCP3008(0)

while True:
    val = emg.value * 3.3
    print(val)
    sleep(1)
