import sounddevice as sd
import numpy as np
import datetime

# Configurar los parámetros de grabación
duration = 10  # Duración de cada grabación en segundos
sample_rate = 44100  # Tasa de muestreo en Hz

# Función para guardar la grabación en un archivo WAV
def save_recording(frames):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    sd.write(filename, frames, sample_rate)

# Función de callback para la grabación
def callback(indata, frames, time, status):
    if status:
        print('Error:', status)
    save_recording(indata.copy())


# Iniciar la grabación en bucle
with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
    sd.sleep(int(duration) * 1000)
