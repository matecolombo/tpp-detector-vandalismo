import time
import png
import busio
import board
import digitalio
from adafruit_ov7670 import (
    OV7670,
    OV7670_SIZE_DIV16,
    OV7670_COLOR_YUV,
    OV7670_TEST_PATTERN_COLOR_BAR_FADE,
)

import tempfile

image_path = tempfile.mktemp(suffix='.png')

# Ensure the camera is shut down, so that it releases the SDA/SCL lines,
# then create the configuration I2C bus

with digitalio.DigitalInOut(board.GP10) as shutdown:
    shutdown.switch_to_output(True)
    time.sleep(0.001)
    bus = busio.I2C(board.GP9, board.GP8)

'''
 adafruit_ov7670.OV7670(i2c_bus: I2C, 						# busio.I2C()
                            data_pins: List[Pin], 			# [D0, D1, D2, D3, D4, D5, D6, D7] - List of 8 data pins 
                            clock: Pin,						# PLK - The pixel clock from the OV7670.
                            vsync: Pin,						# VS - The vsync signal from the OV7670.
                            href: Pin,						# HS - The href signal from the OV7670, sometimes inaccurately called hsync.
                            shutdown: Pin | None = None,	# PWDN - If not None, the shutdown signal to the camera, also called the powerdown or enable pin.
                            reset: Pin | None = None,		# RET - If not None, the reset signal to the camera.
                            mclk: Pin | None = None,		# XLK - Master clock signal/ None if the master clock signal is already being generated.
                            mclk_frequency: int = 16000000,	# The frequency of the master clock
                            i2c_address: int = 33			# The I2C address of the camera.
                        )

'''
cam = OV7670(
    bus,
    data_pins=[
        board.GP12,
        board.GP13,
        board.GP14,
        board.GP15,
        board.GP16,
        board.GP17,
        board.GP18,
        board.GP19,
    ],
    clock=board.GP11,
    vsync=board.GP7,
    href=board.GP21,
    mclk=board.GP20,
    shutdown=None,
    reset=board.GP10,
)

# Configuración de la comunicación serie
#uart = busio.UART(board.GP1, board.GP0, baudrate=115200)  # Conexión UART (RX, TX)

# Dentro del bucle while:
#while True:
buf = bytearray(cam.width * cam.height * 2)
cam.capture(buf)

# RGB565 (also known as 16-bit RGB) is a color format that uses 16 bits to represent a color,
# with 5 bits for the red channel, 6 bits for the green channel, and 5 bits for the blue channel. 

# Guardar la imagen en un archivo temporal en la Raspberry Pi
image_data = []
for y in range(cam.height):
    row_data = []
    for x in range(cam.width):
        pixel = buf[(y * cam.width + x) * 2]
        row_data.extend([pixel, pixel, pixel])
    image_data.append(row_data)
image_path = 'E:/Temp_Image/image.png'   # Ruta y nombre de archivo temporal

with open(image_path, 'wb') as file:
    writer = png.Writer(width=cam.width, height=cam.height, greyscale=True)
    writer.write(file, image_data)

'''
# Enviar la imagen a través de la comunicación serie
try:
    with uart as ser:
        with open(image_path, 'rb') as file:
            data = file.read()
            ser.write(data)
        print('Imagen enviada exitosamente.')
except Exception as e:
    print(f'Error al enviar la imagen: {str(e)}')

# ...

#  time.sleep(0.05)
'''