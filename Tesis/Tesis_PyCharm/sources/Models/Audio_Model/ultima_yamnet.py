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


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform



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
        if i == 600:
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



# Definir el tamaño del conjunto de evaluación y prueba
eval_size = int(0.2 * len(all_data))  # Por ejemplo, usar el 20% para evaluación
test_size = int(0.2 * len(all_data))  # Por ejemplo, usar el 20% para prueba

# Dividir el dataset en conjuntos de evaluación, prueba y entrenamiento
eval_ds = spectrogram_ds.take(eval_size)
remaining_ds = spectrogram_ds.skip(eval_size)
test_ds = remaining_ds.take(test_size)
train_ds = remaining_ds.skip(test_size)


# Realizar un shuffle de los datos
buffer_size = len(all_data)  # El tamaño del buffer de mezcla, puedes ajustarlo según tu preferencia
spectrogram_ds = spectrogram_ds.shuffle(buffer_size)

# spectrogram_ds = non_scream_data.map(
#     get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# spectrogram_ds2 = scream_data.map(
#     get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# Mostrar el espectrograma en un plot
num_plots = 10  # Número de espectrogramas que deseas plotear

# Usamos take(num_plots) para obtener solo los primeros num_plots elementos del dataset
# for spectrogram, label in spectrogram_ds.take(num_plots):
#     # Convierte el tensor spectrogram a un array numpy
#     spectrogram_array = spectrogram.numpy()

#     # Plotear el espectrograma usando matplotlib
#     plt.imshow(spectrogram_array[:, :, 0], cmap='viridis', aspect='auto', origin='lower')
#     plt.colorbar(label='Magnitud')
#     plt.xlabel('Tiempo')
#     plt.ylabel('Frecuencia')
#     plt.title('Espectrograma - Etiqueta: {}'.format(label.numpy()))
#     plt.show()


from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


# Imprimir la forma (shape) de spectrogram_ds
for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)

# Obtener el número de etiquetas (comandos)
num_labels = 2

# Normalizar los datos
norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

# Definir el modelo
model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32), 
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
])

model.summary()


base_model = keras.applications.MobileNetV2(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(32, 32, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=input_shape)
x = inputs

x = preprocessing.Resizing(32, 32)(x)
x = norm_layer(x)

x = base_model(x, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(num_labels)(x)
model = keras.Model(inputs, outputs)

model.summary()

# Compilar el modelo
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

# Entrenar el modelo con el dataset
batch_size = 2
epochs = 20
AUTOTUNE = tf.data.AUTOTUNE




for spectrogram, _ in train_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)


train_ds = train_ds.batch(batch_size)
train_ds = train_ds.cache().prefetch(AUTOTUNE)
model.fit(
   train_ds,
#    validation_data=eval_ds,
   epochs=epochs,
   callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),

   )
