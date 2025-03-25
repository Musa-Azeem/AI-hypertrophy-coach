import spidev
import time

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Open SPI bus 0, device (CS) 0
spi.max_speed_hz = 1000000  # 1 MHz
# spi.max_speed_hz = 1350000  # 1.35 MHz

def read_adc(channel):
    """Read SPI data from MCP3008 (0-7 channel)"""
    if channel < 0 or channel > 7:
        return -1  # Invalid channel
    # MCP3008 expects a 3-byte sequence: 
    # 1st byte: Start bit (1)
    # 2nd byte: (Single-ended mode << 7) | (Channel << 4)
    # 3rd byte: Don't care (0x00)
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    
    # Convert response: 10-bit value from MCP3008
    value = ((adc[1] & 3) << 8) | adc[2]
    return value

# Sampling at 1000 Hz
sample_rate = 1000  # Hz
interval = 1.0 / sample_rate  # Time per sample
start_time = time.perf_counter()

try:
    with open('recordings2/curl.csv', 'w') as f:
        f.write(f'time,emg\n')
        while True:
            loop_start = time.perf_counter()

            channel_0_value = read_adc(0)  # Read channel 0
            f.write(f'{time.time()},{channel_0_value}\n')
            print(channel_0_value / 1023 * 5)
            
            # time.sleep(1/1000)
            elapsed_time = time.perf_counter() - loop_start
            sleep_time = max(0, interval - elapsed_time)
            time.sleep(sleep_time)
except KeyboardInterrupt:
    print("Stopping...")
    spi.close()
