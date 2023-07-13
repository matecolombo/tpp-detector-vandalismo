import machine
import time
from machine import Pin, I2C
from time import sleep_ms
import global_variables

# Resetear IMU cuando este deje de transmitir
def reset_IMU():
    # Ejecutar este código cuando el IMU deje de transmitir.

    # Definir las direcciones de los registros del MPU 6050
    PWR_MGMT_1 = 0x6B
    WHO_AM_I = 0x75

    # Inicializar el bus I2C
    i2c = machine.I2C(0, sda=machine.Pin(8), scl=machine.Pin(9), freq=400000)  # Utilizar pines 8 y 9

    # Dirección del MPU 6050
    mpu6050_addr = 0x68

    # Leer el valor del registro WHO_AM_I para verificar la comunicación
    data = i2c.readfrom_mem(mpu6050_addr, WHO_AM_I, 1)
    print("Valor del registro WHO_AM_I: ", data[0])

    # Configurar el registro PWR_MGMT_1 para activar el MPU 6050
    i2c.writeto_mem(mpu6050_addr, PWR_MGMT_1, b'\x00')

# Normalizar datos de aceleración
def normalize_accel(data):
    return data/16384.0

# Normalizar datos de velocidad angular
def normalize_gyro(data):
    return data/ 131.0

# Calcular aceleración en m/s^2
def calculate_acceleration(raw_data):
    accel_x = normalize_accel(raw_data[0] << 8 | raw_data[1])
    accel_y = normalize_accel(raw_data[2] << 8 | raw_data[3])
    accel_z = normalize_accel(raw_data[4] << 8 | raw_data[5])
    return accel_x, accel_y, accel_z

# Calcular velocidad angular en rad/s
def calculate_angular_velocity(raw_data):
    gyro_x = normalize_gyro(raw_data[0] << 8 | raw_data[1])
    gyro_y = normalize_gyro(raw_data[2] << 8 | raw_data[3])
    gyro_z = normalize_gyro(raw_data[4] << 8 | raw_data[5])
    return gyro_x, gyro_y, gyro_z


def calculate_temperature(raw_data):
    temp = ((raw_data[0] << 8) | raw_data[1]) / 340 + 36.53
    return temp

# Leer valores de aceleración y velocidad angular
def read_sensor_data(addr,ax_dir, temp_dir):
    raw_data = i2c.readfrom_mem(addr, ax_dir, 14)
    accel_data = calculate_acceleration(raw_data[0:6])
    temp_data = calculate_temperature(i2c.readfrom_mem(addr, temp_dir, 2))
    gyro_data = calculate_angular_velocity(raw_data[8:14])

    # Corroborar que no sea nula la medición
    if (all(value == 0 for value in accel_data))&(all(value == 0 for value in gyro_data)) :
        reset_IMU()

    return accel_data, gyro_data, temp_data

# Detectar movimiento y golpes
def detect_motion_temperature_and_impact(th_motion, th_impact, th_tempaddr,ax_dir, temp_dir):
    accel_prev = read_sensor_data(addr,ax_dir, temp_dir)[0]  # Lectura inicial
    motion_status = 0
    movement_status = 0
    temp_status = 0
    while True:
        accel_data, gyro_data, temp_data = read_sensor_data(addr,ax_dir, temp_dir)

        # Detección de movimiento
        diff = abs(accel_data[0] - accel_prev[0]) + \
               abs(accel_data[1] - accel_prev[1]) + \
               abs(accel_data[2] - accel_prev[2])

        # Detección de golpes
        impact = abs(gyro_data[0]) + abs(gyro_data[1]) + abs(gyro_data[2])

        if diff > th_motion:
            motion_status = 1

        if impact > th_impact:
            movement_status = 1

        if temp_data > th_temp:
            temp_status = 1

        accel_prev = accel_data  # Actualizar lectura previa
        sleep_ms(100)  # Esperar antes de la siguiente lectura

        print("{:},{:},{:}".format(motion_status, movement_status, temp_status))

        motion_status = 0
        movement_status = 0
        temp_status = 0
    return motion_status, movement_status, temp_status

# Función principal
def detect_vandalism():

    addr = global_variables.DEVICE_ADDR
    ax_dir = global_variables.ACCEL_XOUT_H
    temp_dir = global_variables.TEMP_H
    th_motion = global_variables.threshold_motion
    th_impact = global_variables.threshold_impact
    th_temp = global_variables.threshold_temp

    # Configuración de I2C
    i2c = machine.I2C(0, scl=machine.Pin(9), sda=machine.Pin(8), freq=400000)# freq=203400 implica Baudrate de 38400 #, freq=400000)

    # Configurar sensor en modo activo
    i2c.writeto_mem(global_variables.DEVICE_ADDR, 0x6B, bytearray([0]))  # PWR_MGMT_1

    # Detectar movimiento y golpes
    detect_motion_temperature_and_impact(th_motion, th_impact, th_temp, addr,ax_dir, temp_dir)

    return motion_status, movement_status, temp_status




