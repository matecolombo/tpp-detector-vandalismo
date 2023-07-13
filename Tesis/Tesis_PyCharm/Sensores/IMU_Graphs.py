'''

import serial

# Configura el puerto y la velocidad de comunicación serial
puerto = 'COM3'  # Ajusta el nombre del puerto según tu configuración
velocidad = 115200  # Ajusta la velocidad según tu configuración
timeout = 1  # Ajusta el tiempo de espera según tus necesidades

# Abre la conexión serial
ser = serial.Serial(puerto, velocidad, timeout=timeout)

# Lee y muestra el log
while True:
    linea = ser.readline().decode().strip()
    print(linea)
'''


import serial

# Configuración de la comunicación serie
port = 'COM8'  # Puerto USB utilizado para la conexión
baudrate = 115200  # Velocidad de comunicación en baudios

# Establecer conexión serie con la Raspberry Pi
try:
    with serial.Serial(port, baudrate) as ser:
        received_data = ser.read_all()
        with open('imagen_recibida.jpg', 'wb') as file:
            file.write(received_data)
        print('Imagen recibida exitosamente.')
except Exception as e:
    print(f'Error al recibir la imagen: {str(e)}')
