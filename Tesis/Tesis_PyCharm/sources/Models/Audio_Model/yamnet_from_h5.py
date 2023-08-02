import tensorflow as tf
from yamnet import pad_waveform, waveform_to_log_mel_spectrogram_patches, ensure_sample_rate
import os
import yamnet
from scipy.io import wavfile
import params
import numpy as np

params = params.Params()

def load_model():
   
  class YAMNet(tf.Module):
    """A TF2 Module wrapper around YAMNet."""
    def __init__(self, weights_path, params):
      super().__init__()
      self._yamnet = yamnet.yamnet_frames_model(params)
      self._yamnet.load_weights(weights_path)
      self._class_map_asset = tf.saved_model.Asset('yamnet_class_map.csv')

    @tf.function(input_signature=[])
    def class_map_path(self):
      return self._class_map_asset.asset_path

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, waveform):
      predictions, embeddings, log_mel_spectrogram = self._yamnet(waveform)

      return {'predictions': predictions,
              'embeddings': embeddings, 
              'log_mel_spectrogram': log_mel_spectrogram}

  # Carga del modelo Yamnet pre-entrenado
  # yamnet_model = tf.keras.models.load_weights('./yamnet.h5')

  
  yamnet_model = YAMNet("yamnet.h5", params)._yamnet

  return yamnet_model



yamnet_model = load_model()
yamnet_model.summary()

# Obtener la última capa de salida del modelo Yamnet
yamnet_output = yamnet_model.layers[-3].output

# Agregar capas adicionales para la nueva tarea (gritos vs. no gritos)
# Puedes experimentar con la arquitectura de estas capas según tus necesidades.
# Aquí hay un ejemplo básico usando una capa densa seguida de una capa de salida.
tf.compat.v1.disable_eager_execution()
# Assuming yamnet_output is the input tensor for your custom layers
# Assuming yamnet_output is the input tensor for your custom layers
model_hijo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=yamnet_output.shape[1:]),
    tf.keras.layers.Dropout(0.5),  # To avoid overfitting
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output layer (screams or no screams)
])

# Create the child model
# model_hijo = tf.keras.models.Model(inputs=yamnet_model.input, outputs=output)

# Congelar todas las capas del modelo padre para que no se actualicen durante el entrenamiento
for layer in yamnet_model.layers:
    layer.trainable = False


model_hijo.summary()
# Compilar el modelo hijo para la nueva tarea
model_hijo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Ahora puedes usar el modelo hijo para entrenar con tus datos etiquetados de gritos y no gritos.
# Asegúrate de que tus datos de entrenamiento estén en el formato adecuado para la entrada del modelo Yamnet.
# Por ejemplo, puedes usar la misma longitud de ventana que se usó durante el entrenamiento del modelo Yamnet para extraer características.



# ------------- Define los hiperparámetros para el modelo y el entrenamiento -------------

non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'

labels = []
features_list = []


# waveform = wavfile.read('./AUDIOS/audio_grito.wav', )[1].astype(np.float32)
# wav_file_name = './AUDIOS/audio_grito.wav'
# sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# # Crear el modelo modificado para detección de gritos usando la función create_modified_yamnet.
# wav_data = wav_data / tf.int16.max
# waveform_padded = pad_waveform(waveform, params)
# log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(waveform_padded, params)

# # Definir la función de normalización
# def normalize_waveform(waveform, max_value):
#     return waveform / max_value



i=0
# Cargar los archivos no gritos y etiquetarlos como 0
for filename in os.listdir(non_scream_dir):
    i+=1
    filepath = os.path.join(non_scream_dir, filename)
    sample_rate, wav_data = wavfile.read(filepath, 'rb')
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    wav_data = tf.cast(wav_data, tf.float32)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    wav_data = wav_data / tf.reduce_max(tf.abs(wav_data))
    print(wav_data.shape)
    waveform_padded = pad_waveform(wav_data, params)
    log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(waveform_padded, params)
    print(features.shape)
    # Realizar cualquier preprocesamiento adicional que necesites en los datos (por ejemplo, cambiar la duración)
    features_list.append(features)
    labels.append(0)
    if i == 5:
        break

i=0
# # Cargar los archivos de gritos y etiquetarlos como 1
for filename in os.listdir(scream_dir):
    i+=1
    filepath = os.path.join(scream_dir, filename)
    sample_rate, wav_data = wavfile.read(filepath, 'rb')
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    wav_data = tf.cast(wav_data, tf.float32)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    wav_data = wav_data / tf.reduce_max(tf.abs(wav_data))
    print(wav_data.shape)
    waveform_padded = pad_waveform(wav_data, params)
    log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(waveform_padded, params)

    print(features.shape)

    # Realizar cualquier preprocesamiento adicional que necesites en los datos (por ejemplo, cambiar la duración)
    features_list.append(features)
    labels.append(1)
    if i == 5:
        break

# print(len(features_list))

# filtered_features_list = [tensor for tensor in features_list]

# Print the contents of filtered_features_list and labels for investigation
# print("Filtered Features List:")
# for i, tensor in enumerate(filtered_features_list):
#     print(f"Sample {i}: {tensor.shape}")


# # Convert lists to TensorFlow tensors
# features_tf = tf.convert_to_tensor(filtered_features_list)
# labels_tf = tf.convert_to_tensor(labels)

# with tf.compat.v1.Session() as sess:
#     features_np = sess.run(features_tf)
#     labels_np = sess.run(labels_tf)


# Print the shapes of both arrays
# print("Features shape:", features_np.shape)
# print("Labels shape:", labels_np.shape)
# # Convertir las listas en arreglos numpy
# features = np.array(features_list)
# labels = np.array(labels)

# print(features[0])
# print(labels[0])

# print(features.shape) 
# print(labels.shape)  

# features = tf.convert_to_tensor(features_list)
# labels = tf.convert_to_tensor(labels)

from sklearn.model_selection import train_test_split

# print(len(labels))
# print(features_list)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(features_list, labels, test_size=0.2, random_state=42)

num_epochs = 30
batch_size = 2
# tf.compat.v1.disable_eager_execution()

# Calculate the number of steps per epoch
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

# Entrena el modelo hijo con tus datos, specifying steps_per_epoch
model_hijo.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)

