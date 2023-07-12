
# IMU - Acelerómetro y giroscopio

from machine import I2C, Pin
import utime
import math

# I2C bus object
i2c = machine.I2C(0, scl=machine.Pin(9), sda=machine.Pin(8), freq=400000)# freq=203400 implica Baudrate de 38400 #, freq=400000) 

# MPU6050 constants
MPU6050_ADDR = 0x68
MPU6050_SMPLRT_DIV = 0x19
MPU6050_CONFIG = 0x1A
MPU6050_GYRO_CONFIG = 0x1B
MPU6050_ACCEL_CONFIG = 0x1C
MPU6050_TEMP = 0x41
MPU6050_ACCEL_XOUT = 0x3B
MPU6050_ACCEL_YOUT = 0x3D
MPU6050_ACCEL_ZOUT = 0x3F
MPU6050_GYRO_XOUT = 0x43
MPU6050_GYRO_YOUT = 0x45
MPU6050_GYRO_ZOUT = 0x47


# MPU6050 initialization
i2c.writeto_mem(MPU6050_ADDR, MPU6050_SMPLRT_DIV, bytearray([0x07]))
i2c.writeto_mem(MPU6050_ADDR, MPU6050_CONFIG, bytearray([0x00]))
i2c.writeto_mem(MPU6050_ADDR, MPU6050_GYRO_CONFIG, bytearray([0x08]))
i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytearray([0x00]))

# Variables de tiempo
t0 = utime.ticks_us()
tiempo_anterior = 0

# Variables de posición, velocidad y aceleración
posicion_anterior = 0.0
velocidad_anterior = 0.0
aceleracion_anterior = 0.0

# Reading acceleration and temperature data
while True:
    # Calcular el tiempo transcurrido desde la última medición
    tiempo_actual = utime.ticks_us()
    delta_tiempo = utime.ticks_diff(tiempo_actual, t0)
    tiempo = delta_tiempo / 1000000  # Convertir a segundos
    t0 = tiempo_actual
    
    accel_x_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_ACCEL_XOUT, 2)
    accel_y_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_ACCEL_YOUT, 2)
    accel_z_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_ACCEL_ZOUT, 2)
    
    gx_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_GYRO_XOUT, 2)
    gy_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_GYRO_YOUT, 2)
    gz_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_GYRO_ZOUT, 2)
    
    temp_raw = i2c.readfrom_mem(MPU6050_ADDR, MPU6050_TEMP, 2)
    
    # Conversión de unidades 
    accel_x = (accel_x_raw[0] << 8 | accel_x_raw[1]) / 16384.0
    accel_y = (accel_y_raw[0] << 8 | accel_y_raw[1]) / 16384.0
    accel_z = (accel_z_raw[0] << 8 | accel_z_raw[1]) / 16384.0
    accel_total = math.sqrt((accel_x ** 2) + (accel_y ** 2) + (accel_z ** 2)) #- 9.8
    
    
    gx = (gx_raw[0] << 8 | gx_raw[1]) / 16384.0
    gy = (gy_raw[0] << 8 | gy_raw[1]) / 16384.0
    gz = (gz_raw[0] << 8 | gz_raw[1]) / 16384.0
    
    temp = ((temp_raw[0] << 8) | temp_raw[1]) / 340 + 36.53
    
    #print("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(accel_x, accel_y, accel_z, gx, gy, gz, temp))
    #print("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(accel_x, accel_y, accel_z, gx, gy, gz))
    print("{:.2f},{:.2f},{:.2f}".format(accel_x, accel_y, accel_z))
    #print("{:.2f},{:.2f},{:.2f}".format(gx, gy, gz))
    # Calcular la velocidad y la posición (según  accel_total)
    velocidad = velocidad_anterior + aceleracion_anterior * tiempo + 0.5 *  accel_total * tiempo**2
    posicion = posicion_anterior + velocidad_anterior * tiempo + 0.5 * aceleracion_anterior * tiempo**2
    
    # Actualizar las variables anteriores para la próxima iteración
    aceleracion_anterior =  accel_total
    velocidad_anterior = velocidad
    posicion_anterior = posicion
    
    #print("{:.2f},{:.2f},{:.2f}".format(accel_total, velocidad, posicion))

    '''
    # Imprimir los resultados
    print("Temperatura: {:.2f} C".format(temp))
    print("Aceleración: aX={:.2f}, aY={:.2f}, aZ={:.2f}, a={:.2f} g".format(accel_x, accel_y, accel_z,accel_total))
    print("Giroscopio: gX={:.2f}, gY={:.2f}, gZ={:.2f} deg".format(gx, gy, gz))
    print("Velocidad: {:.2f} m/s".format(velocidad))
    print("Posición: {:.2f} m".format(posicion))
    '''
    
    # Wait for 1 second before reading data again
    utime.sleep(1)
