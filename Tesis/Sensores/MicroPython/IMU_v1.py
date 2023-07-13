import machine
import utime
import math

i2c = machine.I2C(0, scl=machine.Pin(9), sda=machine.Pin(8))
mpu_addr=0x68#|0x69

aux = bytes([0]) #aux = b'\x00'

i2c.writeto_mem(mpu_addr, 0x6E,aux.decode('utf-8'))

def read_accelerometer():
    #data = i2c.readfrom_mem(mpu_addr, 0x6E, 16)
    data = i2c.readfrom_mem(mpu_addr, 0x3B, 6)
    #print(data)
    x = (data[0] << 8) | data[1]
    y = (data[2] << 8) | data[3]
    z = (data[4] << 8) | data[5]
    return (x, y, z)

def normalize(data):
    return data/16384.0

def hypotenuse(a,b):
    return math.sqrt(a**2 + b**2)

def num2degree(num):
    return num* 180.0/math.pi

def inclination(norm_x,norm_y,norm_z):
    pitch = num2degree(math.atan2(norm_x,hypotenuse(norm_y,norm_z)))
    roll = num2degree(math.atan2(norm_y,hypotenuse(norm_x,norm_z)))
    return pitch,roll
    
while True:
    accel = read_accelerometer()
    
    accel_x = normalize(accel[0])
    accel_y = normalize(accel[1])
    accel_z = normalize(accel[2])

    pitch, roll = inclination(accel_x, accel_y, accel_z)
    
    #print("x: {:.2f}, y: {:.2f}, z: {:.2f},Pitch: {:.2f}deg, Roll: {:.2f}deg.".format(accel_x, accel_y, accel_z,pitch, roll))
    print("{:.2f}, {:.2f}, {:.2f}".format(accel_x, accel_y, accel_z))
    utime.sleep(0.5)
                       
                       
                       
                       