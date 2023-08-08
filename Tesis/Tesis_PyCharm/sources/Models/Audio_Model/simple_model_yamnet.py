import os

# from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_io as tfio
import tensorflow as tf
# from yamnet_from_h5 import pad_waveform
from scipy.io import wavfile
import yamnet
import params
import scipy.signal as signal
import numpy as np
import random

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

yamnet_model.summary()

def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


# _ = tf.keras.utils.get_file('esc-50.zip',
#                         'https://github.com/karoldvl/ESC-50/archive/master.zip',
#                         cache_dir='./',
#                         cache_subdir='datasets',
#                         extract=True)
# esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
# base_data_path = './datasets/ESC-50-master/audio/'

# pd_data = pd.read_csv(esc50_csv)
# pd_data.head()

# my_classes = ['dog', 'yell']
# map_class_to_id = {'dog':0, 'yell':1}

# filtered_pd = pd_data[pd_data.category.isin(my_classes)]


# class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
# filtered_pd = filtered_pd.assign(target=class_id)

# full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
# filtered_pd = filtered_pd.assign(filename=full_path)

# filtered_pd.head(20)

# filenames = filtered_pd['filename']
# targets = filtered_pd['target']
# folds = filtered_pd['fold']

# main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
# main_ds.element_spec


def load_wav_for_map(filename, label, fold):
  return load_wav_16k_mono(filename), label, fold

# main_ds = main_ds.map(load_wav_for_map)
# main_ds.element_spec


# applies the embedding extraction model to a wav data
# def extract_embedding(wav_data, label, fold):
#   ''' run YAMNet to extract embedding from the wav data '''
#   scores, embeddings, spectrogram = yamnet_model(wav_data)
#   num_embeddings = tf.shape(embeddings)[0]
#   return (embeddings,
#             tf.repeat(label, num_embeddings),
#             tf.repeat(fold, num_embeddings))

# # extract embedding
# main_ds = main_ds.map(extract_embedding).unbatch()
# main_ds.element_spec


# # Splitting the data

# cached_ds = main_ds.cache()
# train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
# val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
# test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# # remove the folds column now that it's not needed anymore
# remove_fold_column = lambda embedding, label, fold: (embedding, label)

# train_ds = train_ds.map(remove_fold_column)
# val_ds = val_ds.map(remove_fold_column)
# test_ds = test_ds.map(remove_fold_column)

# train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
# val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
# test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)


# my_model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
#                           name='input_embedding'),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(len(2))
# ], name='my_model')

# my_model.summary()


# my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                  optimizer="adam",
#                  metrics=['accuracy'])

# # callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
# #                                             patience=3,
# #                                             restore_best_weights=True)


# history = my_model.fit(train_ds,
#                        epochs=20,
#                        validation_data=val_ds,
#                        callbacks=callback)


# loss, accuracy = my_model.evaluate(test_ds)

# print("Loss: ", loss)
# print("Accuracy: ", accuracy)

# def apply_data_augmentation(audio, label):
#     # Cambio de tono (pitch shift)
#     # pitch_factor = random.uniform(-0.2, 0.2)
#     # audio = tf.signal.(audio, tf.cast(tf.shape(audio)[0] * (2.0 ** pitch_factor), tf.int32))

#     # Modulación de amplitud (amplitude modulation)
#     depth = random.uniform(0.1, 0.3)
#     frequency = random.uniform(0.2, 0.5)
#     audio = audio + depth * tf.math.sin(2 * np.pi * frequency * tf.range(len(audio)))

#     # Desplazamiento del tiempo (time shift)
#     time_shift = tf.random.uniform(shape=[], minval=0, maxval=5000, dtype=tf.int32)
#     audio = tf.roll(audio, time_shift, axis=0)

#     return audio, label



non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'

# Crear una lista con los nombres de archivo y sus etiquetas para cada clase
non_scream_files = os.listdir(non_scream_dir)
scream_files = os.listdir(scream_dir)

non_scream_data = [(os.path.join(non_scream_dir, file), 'non_scream') for file in non_scream_files]
scream_data = [(os.path.join(scream_dir, file), 'scream') for file in scream_files]

# Combinar las listas de datos de ambas clases
data = non_scream_data + scream_data

# Crear un DataFrame con los datos combinados
pd_data = pd.DataFrame(data, columns=['filename', 'category'])

# Mapear las etiquetas a identificadores numéricos
my_classes = ['non_scream', 'scream']
map_class_to_id = {'non_scream': 0, 'scream': 1}
class_id = pd_data['category'].apply(lambda name: map_class_to_id[name])
pd_data = pd_data.assign(target=class_id)

# Crear una columna de folds (pueden ser valores aleatorios en caso de no tener información específica)
fold_values = [1, 2, 3, 4, 5]
pd_data['fold'] = np.tile(fold_values, len(pd_data) // len(fold_values) + 1)[:len(pd_data)]

# Cargar los archivos de audio y crear el dataset
def load_wav_for_map(filename, label, fold):
    audio_binary = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)  # Puedes ajustar los parámetros según tus necesidades
    return audio, label, fold

main_ds = tf.data.Dataset.from_tensor_slices((pd_data['filename'], pd_data['target'], pd_data['fold']))

# Agregar impresión de pantalla para verificar main_ds
# print("main_ds:")
# for item in main_ds.take(5):
#     print(item)
main_ds = main_ds.map(load_wav_for_map)
main_ds.element_spec

    

# applies the embedding extraction model to a wav data
def extract_embedding(wav_data, label, fold):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  num_embeddings = tf.shape(embeddings)[0]
  return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))

# extract embedding
main_ds = main_ds.map(extract_embedding).unbatch()
main_ds.element_spec

cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

# remove the folds column now that it's not needed anymore
remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
# train_ds = train_ds.map(apply_data_augmentation)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)



my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(my_classes))
], name='my_model')

my_model.summary()


# my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                  optimizer="adam",
#                  metrics=['accuracy'])

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
#                                             patience=3,
#                                             restore_best_weights=True)


# for batch_inputs, batch_labels in train_ds:
#     print("Batch Inputs:", batch_inputs)
#     print("Batch Labels:", batch_labels)
#     break  # Solo imprime el primer lote para evitar una gran cantidad de salida

# input_data = []
# for batch_inputs, _ in train_ds:
#     input_data.append(batch_inputs.numpy())
# input_data = np.concatenate(input_data, axis=0)

# print("Mean of input data:", np.mean(input_data))
# print("Standard deviation of input data:", np.std(input_data))


# # Assuming your data is images (change as per your data)
# for batch_inputs, batch_labels in train_ds:
#     for i in range(5):  # Visualize the first 5 samples
#         plt.imshow(batch_inputs[i])
#         plt.title("Label: " + str(batch_labels[i]))
#         plt.show()
#     break  # Only visualize the first batch


# history = my_model.fit(train_ds,
#                        epochs=20,
#                        validation_data=val_ds,
#                        callbacks=callback)


save_model_path = './modelo_hijo'
# # Guardar el modelo entrenado
# my_model.save(save_model_path)

# loss, accuracy = my_model.evaluate(test_ds)

# print("Loss: ", loss)
# print("Accuracy: ", accuracy)


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform

from keras.models import load_model


# Cargar el modelo entrenado
my_model = load_model(save_model_path)


# loss, accuracy = my_model.evaluate(val_ds)
# my_model.summary()

wav_file_name = './AUDIOS/train/train_data/audio_grito.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# wav_data = np.mean(wav_data, axis=1).astype(np.float32)

sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

scores, embeddings, spectrogram = yamnet_model(wav_data)
result = my_model(embeddings).numpy()

print(result)
inferred_class = my_classes[result.mean(axis=0).argmax()]
# print(f'The main sound is: {inferred_class}')

class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)
  

saved_model_path = './gritos_yamnet'
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
import tensorflow_hub as hub

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

result = serving_model(wav_data).numpy()



inferred_class = my_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}')



non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'

labels = []
features_list = []
data_list = []


def preprocess_audio_file(filepath):
    sample_rate, wav_data = wavfile.read(filepath)
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    wav_data = tf.constant(wav_data, dtype=tf.float32)      
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

from sklearn.model_selection import train_test_split

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

serving_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

serving_model.fit(train_data, train_labels, epochs=20, batch_size=2, validation_split=0.1)