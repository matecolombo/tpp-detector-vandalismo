
# MQ - Detector de gases

from machine import ADC, Pin
import utime

pin_sensor = Pin(26, Pin.IN)
adc = ADC(0)

''''
#Configuración
adc.atten(ADC.ATTN_11DB) # Resolución (11dB)
adc.width(ADC.WIDTH_12BIT) # Frecuencia de Muestreo (12 Bit)
'''

while True:
    value = adc.read_u16()
    if value!=0:
        voltage = value* 3.3 / 65535
        print("Tensión: ", voltage, "V\n")
    else:
        print("--\n")
        
    utime.sleep(0.5)