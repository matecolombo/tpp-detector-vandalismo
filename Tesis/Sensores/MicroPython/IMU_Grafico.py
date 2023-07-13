from machine import Pin, I2C
import utime
from utime import sleep_ms
from math import sqrt

#Parameters
CALIB_ITER = 200#2000
PIC_PERIOD_MIN = 10#1000

# MPU6050 Measurement & Filtering Range
AFS_SEL = 2  # Accelerometer Configuration Settings   AFS_SEL=2, Full Scale Range = +/- 8 [g]
DLPF_SEL = 0  # DLPF Configuration Settings  Accel BW 260Hz, Delay 0ms / Gyro BW 256Hz, Delay 0.98ms, Fs 8KHz 

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

# Variables for gravity
Cal_AcX, Cal_AcY, Cal_AcZ = 0, 0, 0  # Calibration values
Min_GAcX, Max_GAcX, PtoP_GAcX = 0, 0, 0
Min_GAcY, Max_GAcY, PtoP_GAcY = 0, 0, 0
Min_GAcZ, Max_GAcZ, PtoP_GAcZ = 0, 0, 0
cnt = 0  # Count of calibration process
Grvt_unit = 0  # Gravity value unit
period, prev_time = 0, 0  # Period of calculation
FREQ=400000

#I2C
i2c = machine.I2C(0, scl=machine.Pin(9), sda=machine.Pin(8), freq=FREQ)

def init_MPU6050():
    # MPU6050 Initializing & Reset
    i2c.writeto_mem(MPU6050_ADDR, 0x6B, bytes([0]))

    # MPU6050 Clock Type
    i2c.writeto_mem(MPU6050_ADDR, 0x6B, bytes([0x03]))
        
    i2c.writeto_mem(MPU6050_ADDR, MPU6050_SMPLRT_DIV, bytearray([0x07]))
    i2c.writeto_mem(MPU6050_ADDR, MPU6050_CONFIG, bytearray([0x00]))
    i2c.writeto_mem(MPU6050_ADDR, MPU6050_GYRO_CONFIG, bytearray([0x08]))
    #i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytearray([0x00]))
    
    # MPU6050 Accelerometer Configuration Setting
    if AFS_SEL == 0:
        i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytes([0x00]))
    elif AFS_SEL == 1:
        i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytes([0x08]))
    elif AFS_SEL == 2:
        i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytes([0x10]))
    else:
        i2c.writeto_mem(MPU6050_ADDR, MPU6050_ACCEL_CONFIG, bytes([0x18]))

    # MPU6050 DLPF(Digital Low Pass Filter)
    if DLPF_SEL == 0:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x00]))
    elif DLPF_SEL == 1:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x01]))
    elif DLPF_SEL == 2:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x02]))
    elif DLPF_SEL == 3:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x03]))
    elif DLPF_SEL == 4:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x04]))
    elif DLPF_SEL == 5:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x05]))
    else:
        i2c.writeto_mem(MPU6050_ADDR, 0x1A, bytes([0x06]))


def gravity_range_option():
    global Grvt_unit
    if AFS_SEL == 0:
        Grvt_unit = 16384
    elif AFS_SEL == 1:
        Grvt_unit = 8192
    elif AFS_SEL == 2:
        Grvt_unit = 4096
    elif AFS_SEL == 3:
        Grvt_unit = 3276.8


def calib_MPU6050():
    global Cal_AcX, Cal_AcY, Cal_AcZ, cnt
    for i in range(CALIB_ITER):
        if i % 200 == 0:
            cnt += 1
            if cnt == 1:
                print("Calculating .")
            else:
                print(".")
        read_data_MPU6050()
        sleep_ms(10)
        Cal_AcX += AcX
        Cal_AcY += AcY
        Cal_AcZ += AcZ

    Cal_AcX /= 2000
    Cal_AcY /= 2000
    Cal_AcZ /= 2000

    print("")
    print("End of Calculation")
    print("Cal_AcX =", Cal_AcX)
    print("Cal_AcY =", Cal_AcY)
    print("Cal_AcZ =", Cal_AcZ)


def read_data_MPU6050():
    global AcX, AcY, AcZ
    i2c.writeto(MPU6050_ADDR, bytes([0x3B]), True)
    data = i2c.readfrom(MPU6050_ADDR, 6)
    AcX = int.from_bytes(data[0:2], 'big', True)
    AcY = int.from_bytes(data[2:4], 'big', True)
    AcZ = int.from_bytes(data[4:6], 'big', True)


def calc_grvt():
    global AcX, AcY, AcZ, GAcX, GAcY, GAcZ, Min_GAcX, Max_GAcX, PtoP_GAcX, Min_GAcY, Max_GAcY, PtoP_GAcY, Min_GAcZ, Max_GAcZ, PtoP_GAcZ
    AcX = AcX - Cal_AcX
    AcY = AcY - Cal_AcY
    AcZ = AcZ - Cal_AcZ
    GAcX = AcX / Grvt_unit
    GAcY = AcY / Grvt_unit
    GAcZ = AcZ / Grvt_unit
    Min_GAcX = min(Min_GAcX, GAcX)
    Max_GAcX = max(Max_GAcX, GAcX)
    PtoP_GAcX = Max_GAcX - Min_GAcX
    Min_GAcY = min(Min_GAcY, GAcY)
    Max_GAcY = max(Max_GAcY, GAcY)
    PtoP_GAcY = Max_GAcY - Min_GAcY
    Min_GAcZ = min(Min_GAcZ, GAcZ)
    Max_GAcZ = max(Max_GAcZ, GAcZ)
    PtoP_GAcZ = Max_GAcZ - Min_GAcZ


def display_grvt():
    global period, prev_time, PtoP_GAcX, PtoP_GAcY, PtoP_GAcZ, Min_GAcX, Max_GAcX, Min_GAcY, Max_GAcY, Min_GAcZ, Max_GAcZ
    print("AcX =", AcX)
    print("AcY =", AcY)
    print("AcZ =", AcZ)
    period = utime.ticks_ms() - prev_time
    if period > PIC_PERIOD_MIN:
        prev_time = utime.ticks_ms()
        print("PeakToPeak X|Y|Z")
        print(PtoP_GAcX, PtoP_GAcY, PtoP_GAcZ)
        Min_GAcX = 0
        Max_GAcX = 0
        Min_GAcY = 0
        Max_GAcY = 0
        Min_GAcZ = 0
        Max_GAcZ = 0

import matplotlib.pyplot as plt

# Lista para almacenar los valores de tiempo y amplitudes
time_values = []
amplitude_x = []
amplitude_y = []
amplitude_z = []

# Límite de la cantidad de valores almacenados
max_values = 1000

# Función para graficar en tiempo real
def plot_realtime():
    # Configuración inicial de la figura y los subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # Configuración de los ejes y etiquetas
    ax1.set_ylabel('Amplitud X')
    ax2.set_ylabel('Amplitud Y')
    ax3.set_ylabel('Amplitud Z')
    ax3.set_xlabel('Tiempo')

    # Actualizar y graficar los datos en tiempo real
    while True:
        # Agregar los nuevos valores a las listas
        time_values.append(len(time_values))
        amplitude_x.append(GAcX)  # Reemplaza 'GAcX' con la variable correspondiente
        amplitude_y.append(GAcY)  # Reemplaza 'GAcY' con la variable correspondiente
        amplitude_z.append(GAcZ)  # Reemplaza 'GAcZ' con la variable correspondiente

        # Verificar si se ha alcanzado el límite de valores almacenados
        if len(time_values) > max_values:
            # Eliminar los valores más antiguos
            del time_values[0]
            del amplitude_x[0]
            del amplitude_y[0]
            del amplitude_z[0]

        # Limpiar los subplots y graficar los datos actualizados
        ax1.cla()
        ax1.plot(time_values, amplitude_x)
        ax1.set_ylabel('Amplitud X')

        ax2.cla()
        ax2.plot(time_values, amplitude_y)
        ax2.set_ylabel('Amplitud Y')

        ax3.cla()
        ax3.plot(time_values, amplitude_z)
        ax3.set_ylabel('Amplitud Z')
        ax3.set_xlabel('Tiempo')

        # Actualizar la figura
        plt.tight_layout()
        plt.pause(0.01)



def setup():
    global cnt
    init_MPU6050()
    cnt = 0
    gravity_range_option()
    calib_MPU6050()


def loop():
    read_data_MPU6050()
    calc_grvt()
    display_grvt()
    # Llamar a la función para comenzar a graficar en tiempo real
    plot_realtime()


setup()
while True:
    loop()
