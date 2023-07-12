
# Micrófono

from machine import ADC, Pin
import utime

digital_microphone = machine.Pin(21, machine.Pin.IN)

while True:
    soundDetected = digital_microphone.value()
    if soundDetected:
        print(1)
        utime.sleep(1)
    else:
        print(0)
        utime.sleep(0.01)