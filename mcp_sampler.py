import spidev
import time
import threading

class Sampler(threading.Thread):
    def __init__(self, mcp_channel=0, sample_rate=1000):
        super().__init__()
        self.channel = channel
        self.sample_rate = sample_rate
        self.interval = 1.0 / sample_rate  # Time per sample
        self.running = False
        
        # SPI setup
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)  # SPI bus 0, device (CS) 0
        self.spi.max_speed_hz = 1000000  # 1 MHz
        
    def read_adc(self):
        """Read ADC value from the given channel (0-7) on MCP3008."""
        assert 0 <= self.channel <= 7, "Channel must be between 0 and 7"
        adc = self.spi.xfer2([1, (8 + self.channel) << 4, 0])  # SPI transaction
        value = ((adc[1] & 3) << 8) | adc[2]  # Convert to 10-bit value
        return value

    def run(self):
        """Run the ADC sampling loop at a fixed rate."""
        self.running = True
        print("Starting ADC sampling thread at 1000 Hz...")

        with open('emg.csv', 'w') as f:
            f.write('time,emg\n')
            while self.running:
                loop_start = time.perf_counter()
                
                # Read ADC value
                value = self.read_adc()
                f.write(f'{time.time()},{value}\n')

                # Maintain precise timing
                elapsed_time = time.perf_counter() - loop_start
                sleep_time = max(0, self.interval - elapsed_time)
                time.sleep(sleep_time)

        print("ADC sampling thread stopped.")

    def stop(self):
        """Stop the sampling loop."""
        self.running = False
        self.join()  # Wait for thread to exit
        self.spi.close()  # Close SPI connection

if __name__ == "__main__":
    sampler = MCP3008Sampler(channel=0, sample_rate=1000)
    sampler.start()

    # Run for 10 seconds, then stop
    time.sleep(5)
    sampler.stop()