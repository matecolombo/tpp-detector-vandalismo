
# Micrófono

from machine import ADC, Pin
import utime
import math

#digital_microphone = machine.Pin(21, machine.Pin.IN)
pin_sensor = Pin(26, Pin.IN)
analog_microphone = ADC(0)
sensibilidad = 1 #[mV/Pa]
Vmax = 5.0
P0 = 20 #Presión sonora de referencia [uPa]
dBu_th = 70

while True:
    value = analog_microphone.read_u16() #/65535
    
    #Tensión
    V = value * Vmax/ 1023.0
    
    #Presión sonora
    P = (V*1000) / sensibilidad #[Pa]
    dBu = 20 * math.log(P / P0,10)
    print(dBu)
    
    if (dBu>dBu_th):
        print('Alerta')
        
    utime.sleep(0.1)