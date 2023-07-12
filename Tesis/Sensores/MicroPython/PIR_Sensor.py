##PIR HC-SR501 - Detector de movimiento
import machine
from machine import Pin
import utime

def detect_presence():
    pirPin = Pin(21, machine.Pin.IN) # Input for HC-S501
    pirValue = 0 # Mediciones del PIR
    timeSpan = 2 # [seg]
    timeStep = 1 # [mseg]

    wait = timeStep/1000
    numSamples = round(timeSpan/wait)
    utime.sleep(0.5)

    sumValue = 0
    for i in range(0,numSamples):
        pirValue = pirPin.value()
        sumValue = sumValue + pirValue
        utime.sleep(wait)
    if((((sumValue/numSamples-0.33)*10000)>=0)):
       state = True
    else:
        state = False

    return state

