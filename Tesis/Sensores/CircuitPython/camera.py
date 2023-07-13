# https://docs.circuitpython.org/projects/ov7670/en/latest/examples.html
# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2021 Jeff Epler for Adafruit Industries
#
# SPDX-License-Identifier: Unlicense

"""Capture an image from the camera and display it as ASCII art.

The camera is placed in YUV mode, so the top 8 bits of each color
value can be treated as "greyscale".

It's important that you use a terminal program that can interpret
"ANSI" escape sequences.  The demo uses them to "paint" each frame
on top of the prevous one, rather than scrolling.
"""

import sys
import time

import digitalio
import busio
import board

from adafruit_ov7670 import (  # pylint: disable=unused-import
    OV7670,
    OV7670_SIZE_DIV16,
    OV7670_COLOR_YUV,
    OV7670_TEST_PATTERN_COLOR_BAR_FADE,
)

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
cam.size = OV7670_SIZE_DIV16
cam.colorspace = OV7670_COLOR_YUV
cam.flip_y = True
#cam.test_pattern = OV7670_TEST_PATTERN_COLOR_BAR_FADE

buf = bytearray(2 * cam.width * cam.height)
chars = b" .:-=+*#%@"

width = cam.width
row = bytearray(2 * width)

sys.stdout.write("\033[2J") #Se borra la pantalla de la terminal
#while True:
for i in range (0,2):
    cam.capture(buf)
    for j in range(cam.height): #Se itera sobre cada fila de la imagen capturada.
        sys.stdout.write(f"\033[{j}H") #Se escribe la fila en la terminal y escape
        for i in range(cam.width):
            row[i * 2] = row[i * 2 + 1] = chars[
                buf[2 * (width * j + i)] * (len(chars) - 1) // 255]
        sys.stdout.write(row)
        sys.stdout.write("\033[K")
    sys.stdout.write("\033[J")
    time.sleep(0.5)
