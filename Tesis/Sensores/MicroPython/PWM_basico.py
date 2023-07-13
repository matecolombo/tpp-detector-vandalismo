import time
from machine import Pin, PWM

pwm = PWM(Pin(22))

pwm.freq(1000)

while True:
    for i in range (0,65535,5):
        pwm.duty_u16(i)
        time.sleep(0.01)
    
    print("100% - ON")
    time.sleep(2)
    
    for i in range (65535, 0, -5):
        pwm.duty_u16(i)
        time.sleep(0.01)
    
    print("0% - OFF")
    time.sleep(2)
        
