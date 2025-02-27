# https://randomnerdtutorials.com/raspberry-pi-analog-inputs-python-mcp3008/

# enable spi: sudo raspi-config nonint do_spi 0

from gpiozero import MCP3008
from time import sleep

# refers to MCP3008 channel 0
pot = MCP3008(1)

while True:
    print(pot.value)
    sleep(1)