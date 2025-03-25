import smbus
import time

import smbus
import time

# MPU6050 I2C address
MPU6050_ADDR = 0x68

# Register addresses
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Initialize I2C bus
bus = smbus.SMBus(1)

# Wake up MPU6050 (it starts in sleep mode)
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_word(adr):
    high = bus.read_byte_data(MPU6050_ADDR, adr)
    low = bus.read_byte_data(MPU6050_ADDR, adr + 1)
    value = (high << 8) + low
    if value >= 0x8000:  # Convert to signed
        value = -((65535 - value) + 1)
    return value

HZ = 100
interval = 1.0 / HZ
with open('imu.csv', 'w') as f:
    f.write('time\n')
    while True:
        loop_start = time.perf_counter()

        # Read Accelerometer data
        accel_x = read_word(ACCEL_XOUT_H) / 16384.0
        accel_y = read_word(ACCEL_XOUT_H + 2) / 16384.0
        accel_z = read_word(ACCEL_XOUT_H + 4) / 16384.0

        # Read Gyroscope data
        gyro_x = read_word(GYRO_XOUT_H) / 131.0
        gyro_y = read_word(GYRO_XOUT_H + 2) / 131.0
        gyro_z = read_word(GYRO_XOUT_H + 4) / 131.0

        # print(f"Accel X: {accel_x} g, Y: {accel_y} g, Z: {accel_z} g")
        # print(f"Gyro X: {gyro_x} deg/s, Y: {gyro_y} deg/s, Z: {gyro_z} deg/s")
        # print("-" * 40)

        f.write(f'{time.time()}\n')
        print(accel_x)

        elapsed_time = time.perf_counter() - loop_start
        sleep_time = max(0, interval - elapsed_time)
        time.sleep(sleep_time)