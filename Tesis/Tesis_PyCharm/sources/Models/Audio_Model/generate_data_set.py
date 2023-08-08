import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from tensorflow import keras as keras
import tensorflow as tf
import numpy as np
from scipy.io import wavfile

non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'



def get_spectrogram(waveform, target_length):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([target_length] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram


def get_spectrogram_and_label_id(spectrogram, label):
    spectrogram = tf.expand_dims(spectrogram, -1)
    # Usa la siguiente línea si estás utilizando modelos preentrenados que esperan imágenes en color (3 canales RGB)
    #spectrogram = tf.image.grayscale_to_rgb(spectrogram)

    label_id = tf.argmax(tf.cast(label == 1, tf.int32))  # Compara el valor de label con 1 para determinar si es un grito (scream) o no (non_scream)
    return spectrogram, label_id


def preprocess_audio_file(filepath, target_length):
    _, wav_data = wavfile.read(filepath)
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    waveform_tensor = tf.convert_to_tensor(wav_data, dtype=tf.float32)
    # print(tf.shape(waveform_tensor)[0])
    waveform_length = tf.shape(waveform_tensor)[0]
    
    if waveform_length > target_length:
        # Recortar el waveform si es más largo que target_length
        start_idx = (waveform_length - target_length) // 2
        end_idx = start_idx + target_length
        waveform = waveform_tensor[start_idx:end_idx]
    else:
        # Rellenar con ceros si es más corto que target_length
        zero_padding = tf.zeros([target_length - waveform_length], dtype=tf.float32)
        waveform = tf.concat([waveform_tensor, zero_padding], 0)

    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)


    # spectrogram = get_spectrogram(waveform_tensor, target_length)
    return spectrogram



def load_data(data_dir, label, target_length=None):
    i= 0
    data = []
    labels = []
    target_length = 160000
    for filename in os.listdir(data_dir):
        i = i + 1
        print(i)
        filepath = os.path.join(data_dir, filename)
        print(filepath)
        wav_data = preprocess_audio_file(filepath, target_length)
        spectrogram = tf.expand_dims(wav_data, -1)
        spectrogram_rgb = tf.image.grayscale_to_rgb(spectrogram)
        data.append(spectrogram_rgb)
        labels.append(label)
        if i == 1000:
            break
    # labels = [label] * len(data)
    return data, labels

non_scream_data, non_scream_labels = load_data(non_scream_dir, 0)
scream_data, scream_labels = load_data(scream_dir, 1)

# Unificar los datos y las etiquetas
all_data = non_scream_data + scream_data
all_labels = non_scream_labels + scream_labels

# Crear el dataset usando tf.data.Dataset.from_tensor_slices
spectrogram_ds = tf.data.Dataset.from_tensor_slices((all_data, all_labels))



# Definir una función para convertir un ejemplo en un formato compatible con TFRecord
def serialize_example(feature, label):
    feature = tf.io.serialize_tensor(feature)
    example = tf.train.Example(features=tf.train.Features(feature={
        'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.numpy()])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))
    return example.SerializeToString()

# Definir el nombre del archivo TFRecord
tfrecord_filename = 'spectrogram_dataset.tfrecord'

# Crear un escritor de TFRecord
with tf.io.TFRecordWriter(tfrecord_filename) as writer:
    for spectrogram, label in spectrogram_ds:
        serialized_example = serialize_example(spectrogram, label)
        writer.write(serialized_example)



# Definir la función de análisis de ejemplo
def parse_example(example_proto):
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = example['label']
    return feature, label

# Cargar el dataset desde el archivo TFRecord
tfrecord_dataset = tf.data.TFRecordDataset([tfrecord_filename])
parsed_dataset = tfrecord_dataset.map(parse_example)

dataset_list = list(parsed_dataset)
dataset_size = len(dataset_list)

print("Cantidad de elementos en el dataset:", dataset_size)

# Definir el tamaño del conjunto de evaluación y prueba
eval_size = int(0.2 * dataset_size)  # Por ejemplo, usar el 20% para evaluación
test_size = int(0.2 * dataset_size)  # Por ejemplo, usar el 20% para prueba

# Dividir el dataset en conjuntos de evaluación, prueba y entrenamiento
eval_ds = parsed_dataset.take(eval_size)
remaining_ds = parsed_dataset.skip(eval_size)
test_ds = remaining_ds.take(test_size)
train_ds = remaining_ds.skip(test_size)


for spectrogram, _ in train_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)

