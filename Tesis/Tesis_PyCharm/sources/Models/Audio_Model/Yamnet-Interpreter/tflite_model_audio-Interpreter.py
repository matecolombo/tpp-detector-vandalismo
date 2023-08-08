import tensorflow as tf
import numpy as np
import io
import csv
from scipy.io import wavfile
import scipy.signal as signal
import tensorflow_hub as hub

model_file = "lite-model_yamnet_tflite_1.tflite"

# # Cargar el modelo TFLite
# interpreter = tf.lite.Interpreter(model_path=model_file)
# interpreter.allocate_tensors()


# # Crear un nuevo TFLiteConverter desde el modelo cargado
# converter = tf.lite.TFLiteConverter.from_saved_model(model_file)

# def model_to_TFlite_Reduced_Float(converter, tflite_file):
#     # converter = tf.lite.TFLiteConverter.from_saved_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_types = [tf.float16]
#     print(converter)
#     tflite_model = converter.convert()
#     open(tflite_file, "wb").write(tflite_model)

# model_to_TFlite_Reduced_Float(converter, 'lite_model_yamnet_tflite_reduced_float.tflite')


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


# Download the model to yamnet.tflite
interpreter = tf.lite.Interpreter('lite-model_yamnet_tflite_1.tflite')
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']
embeddings_output_index = output_details[1]['index']
spectrogram_output_index = output_details[2]['index']

print(waveform_input_index)

# Input: 3 seconds of silence as mono 16 kHz waveform samples.
#waveform = np.zeros(3 * 16000, dtype=np.float32)
waveform = wavfile.read('../AUDIOS/miauing.wav', )[1].astype(np.float32)

wav_file_name = '../AUDIOS/miauing.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)


duracion_fragmento = 10

# Calcular la duración total del audio en segundos
duracion_total_audio = len(wav_data) / sample_rate

# Calcular el número total de fragmentos
num_fragmentos = int(np.ceil(duracion_total_audio / duracion_fragmento))

print(num_fragmentos)
# Fragmentar el audio
fragmentos_de_audio = []
for i in range(num_fragmentos):
    inicio = i * duracion_fragmento * sample_rate
    fin = min((i + 1) * duracion_fragmento * sample_rate, len(wav_data))
    fragmento = wav_data[inicio:fin]
    fragmentos_de_audio.append(fragmento)

for fragmento in fragmentos_de_audio:
    print(fragmento)
    fragmento = fragmento / tf.int16.max
#print(type(waveform))
    interpreter.resize_tensor_input(waveform_input_index, [len(fragmento)])
    print(waveform_input_index)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, fragmento.astype(np.float32))
    interpreter.invoke()
    scores, embeddings, spectrogram = (
        interpreter.get_tensor(scores_output_index),
        interpreter.get_tensor(embeddings_output_index),
        interpreter.get_tensor(spectrogram_output_index))
    print(scores.shape, embeddings.shape, spectrogram.shape)  # (N, 521) (N, 1024) (M, 64)

    # Download the YAMNet class map (see main YAMNet model docs) to yamnet_class_map.csv
    # See YAMNet TF2 usage sample for class_names_from_csv() definition.
    class_names = class_names_from_csv(open('yamnet_class_map.csv').read())
    # Suponiendo que 'scores' es un arreglo de numpy con los puntajes
    mean_scores = scores.mean(axis=0)
    top_indices = np.argsort(mean_scores)[-3:]  # Obtener los índices de los tres máximos

    # Obtener los nombres correspondientes a los índices máximos
    top_class_names = [class_names[i] for i in top_indices]

    # Imprimir los tres nombres correspondientes a los máximos puntajes
    print(top_class_names)
