
import struct
from PIL import Image
import numpy as np


def generate_image_from_file_v1(file_path, width, height):
    with open(file_path, 'rb') as file:
        data = file.read()

    # Decodificar los bytes y dividir en una lista de elementos
    bytes_list = data.decode().split('\r\n')

    # Eliminar elementos vacíos
    bytes_list = list(filter(None, bytes_list))
    #bytes_list = [bytearray.fromhex(element)[::-1].hex() for element in bytes_list]             ########################### CHEQUEAR SI ES NECESARIO
    # Unir los bytes de a dos y separarlos en grupos de 5-6-5 bits
    words = []
    for i in range(0, len(bytes_list), 2):
        byte1 = int(bytes_list[i], 2)
        byte2 = int(bytes_list[i + 1], 2)
        word = ((byte1 << 8) | byte2) & 0xFFFF  # Unir los bytes en una palabra de 16 bits
        group1 = (word >> 11) & 0b11111  # Primeros 5 bits
        group2 = (word >> 5) & 0b111111  # Siguientes 6 bits
        group3 = word & 0b11111  # Últimos 5 bits
        words.append((round(group1 / 31 * 255), round(group2 / 31 * 255), round(group3 / 31 * 255)))

#    pixels = np.array(words).reshape((width, height, 3))
    pixels = np.array(words).reshape((height,width, 3))
    image = Image.fromarray(pixels.astype(np.uint8))
    image.save("output.png", "PNG")

    return image


def generate_image_from_file_v2(file_path, width, height):
    with open(file_path, 'rb') as file:
        data = file.read()

    # Decodificar los bytes y dividir en una lista de elementos
    bytes_list = data.decode().split()
    print(len(bytes_list))
    # Unir los bytes de a dos y separarlos en grupos de 5-6-5 bits
    words = []
    length = width * height * 3
    for i in range(0, len(bytes_list)-1, 2):
        byte1 = int(bytes_list[i], 2)
        byte2 = int(bytes_list[i + 1], 2)
        word = ((byte1 << 8) | byte2) & 0xFFFF  # Unir los bytes en una palabra de 16 bits
        print(len(words))
        group1 = (word >> 11) & 0b11111  # Primeros 5 bits
        group2 = (word >> 5) & 0b111111  # Siguientes 6 bits
        group3 = word & 0b11111  # Últimos 5 bits
        words.append((round(group1 / 31 * 255), round(group2 / 31 * 255), round(group3 / 31 * 255)))


    # Los píxeles están organizados por filas y columnas
    pixels = np.array(words).reshape((height, width, 3))
    image = Image.fromarray(pixels.astype(np.uint8))
    image.save("output.png", "PNG")

    return image

# Ruta del archivo de texto con el código binario RGB565
file_path_v1 = "../../Sensores/CircuitPython/Camera_pics/Binary_pic.txt"
file_path_v2 = "received_data.txt"
# Ancho y alto de la imagen
width = 160#80
height = 120#60

# Generar la imagen desde el archivo
#image = generate_image_from_file_v1(file_path_v1, width, height)
image = generate_image_from_file_v1(file_path_v2, width, height)

# Guardar la imagen como archivo PNG
image.save("../../Sensores/CircuitPython/Camera_pics/picture_test.png", "PNG")
