#
# Ejecutar este código cuando el IMU deje de transmitir. 
#


import machine

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
