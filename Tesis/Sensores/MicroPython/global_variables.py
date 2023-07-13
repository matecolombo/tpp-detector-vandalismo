# Definición de variables globales

# Parámetros de umbral

# IMU
threshold_motion = 0.6  # Umbral de movimiento
threshold_impact = 700  # Umbral de golpe 
threshold_temp = 223    # Umbral de temperatura


# Registros del acelerómetro y giroscopio
DEVICE_ADDR = 0x68 # Dirección I2C del sensor MPU 6050
ACCEL_XOUT_H = 0x3B
ACCEL_XOUT_L = 0x3C
ACCEL_YOUT_H = 0x3D
ACCEL_YOUT_L = 0x3E
ACCEL_ZOUT_H = 0x3F
ACCEL_ZOUT_L = 0x40
TEMP_H = 0x41
TEMP_L = 0x42
GYRO_XOUT_H = 0x43
GYRO_XOUT_L = 0x44
GYRO_YOUT_H = 0x45
GYRO_YOUT_L = 0x46
GYRO_ZOUT_H = 0x47
GYRO_ZOUT_L = 0x48
