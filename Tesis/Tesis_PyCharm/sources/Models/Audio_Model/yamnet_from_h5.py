import tensorflow as tf
from yamnet import pad_waveform, waveform_to_log_mel_spectrogram_patches, ensure_sample_rate
import os
import yamnet
from scipy.io import wavfile
import params
import numpy as np
import tensorflow_hub as hub


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


  yamnet_model = YAMNet("yamnet.h5", params)._yamnet

  return yamnet_model

yamnet_model = load_model()
# yamnet_model.summary()

# Obtener la última capa de salida del modelo Yamnet
yamnet_output = yamnet_model.layers[-3].output

# yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
# yamnet_model = hub.load(yamnet_model_handle)

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model,
                                            trainable=False, name='yamnet')

# intermediate_model = tf.keras.Model(inputs=yamnet_model.input, outputs=yamnet_model.layers[-3].output)

# Congelar todas las capas del modelo padre para que no se actualicen durante el entrenamiento
for layer in yamnet_model.layers:
    layer.trainable = False

# Agregar capas adicionales para la nueva tarea (gritos vs. no gritos)
model_hijo = tf.keras.Sequential([
   tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Para evitar el sobreajuste
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida binaria (gritos o no gritos)
])

model_hijo.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])



# callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
#                                             patience=3,
#                                             restore_best_weights=True)


class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)

_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = model_hijo(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)


# serving_model.save(saved_model_path, include_optimizer=False)
# Crear el modelo completo fusionando el intermediate_model y el model_hijo
# child_model = tf.keras.models.Model(inputs=intermediate_model.input, outputs=model_hijo(intermediate_model.output))

# Congelar todas las capas del modelo padre (intermediate_model) para que no se actualicen durante el entrenamiento
# for layer in intermediate_model.layers:
#     layer.trainable = False

# # Compilar el child_model
# child_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# child_model.summary()






# ------------- Define los hiperparámetros para el modelo y el entrenamiento -------------

non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'

labels = []
features_list = []
data_list = []


def preprocess_audio_file(filepath):
    sample_rate, wav_data = wavfile.read(filepath)
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    wav_data = tf.cast(wav_data, tf.float32)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
    return wav_data


def load_data(data_dir, label):
    data = []
    labels = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        wav_data = preprocess_audio_file(filepath)
        data.append(wav_data)
        labels.append(label)
    return data, labels


non_scream_data, non_scream_labels = load_data(non_scream_dir, 0)  # Label 0 para no gritos
scream_data, scream_labels = load_data(scream_dir, 1)  #


# Combina los datos y las etiquetas
all_data = non_scream_data + scream_data
all_labels = non_scream_labels + scream_labels


# Convierte las listas de Python a arrays de numpy
all_data = np.array(all_data)
all_labels = np.array(all_labels)

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split


# num_classes = 2  # Number of classes (screams and non-screams)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# print(y_train.tolist())

# # Convert features_list and labels to numpy arrays
# X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
# y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)


num_epochs = 30
batch_size = 4

# Suponiendo que tienes ya definido el modelo (model) y compilado con una función de pérdida apropiada, etc.
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# # Calculate the number of steps per epoch
# steps_per_epoch = int(np.ceil(len(X_train) / batch_size))


# # Entrena el modelo hijo con tus datos, specifying steps_per_epoch
# model_hijo.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch)