import PIR_Sensor
import IMU_v4


while True:
    #PIR - HC-SR501
#    presence_status = PIR_Sensor.detect_presence()
#    print(presence_status)
    
    #IMU - MPU6050
    motion_status, movement_status, temp_status = IMU_v4.detect_vandalism()
    print(motion_status)
    print(movement_status)
    print(temp_status)
    
    