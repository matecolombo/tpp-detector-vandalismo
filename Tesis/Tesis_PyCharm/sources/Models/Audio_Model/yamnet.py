# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda, Dense
from scipy.io import wavfile
import scipy.signal as signal
import io
# import features as features_lib
import params

params = params.Params()

def _batch_norm(name, params):
  def _bn_layer(layer_input):
    return layers.BatchNormalization(
      name=name,
      center=params.batchnorm_center,
      scale=params.batchnorm_scale,
      epsilon=params.batchnorm_epsilon)(layer_input)
  return _bn_layer


def _conv(name, kernel, stride, filters, params):
  def _conv_layer(layer_input):
    output = layers.Conv2D(name='{}/conv'.format(name),
                           filters=filters,
                           kernel_size=kernel,
                           strides=stride,
                           padding=params.conv_padding,
                           use_bias=False,
                           activation=None)(layer_input)
    output = _batch_norm('{}/conv/bn'.format(name), params)(output)
    output = layers.ReLU(name='{}/relu'.format(name))(output)
    return output
  return _conv_layer


def _separable_conv(name, kernel, stride, filters, params):
  def _separable_conv_layer(layer_input):
    output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                    kernel_size=kernel,
                                    strides=stride,
                                    depth_multiplier=1,
                                    padding=params.conv_padding,
                                    use_bias=False,
                                    activation=None)(layer_input)
    output = _batch_norm('{}/depthwise_conv/bn'.format(name), params)(output)
    output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
    output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                           filters=filters,
                           kernel_size=(1, 1),
                           strides=1,
                           padding=params.conv_padding,
                           use_bias=False,
                           activation=None)(output)
    output = _batch_norm('{}/pointwise_conv/bn'.format(name), params)(output)
    output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
    return output
  return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]


def yamnet(features, params):
  """Define the core YAMNet mode in Keras."""
  net = layers.Reshape(
      (params.patch_frames, params.patch_bands, 1),
      input_shape=(params.patch_frames, params.patch_bands))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
  embeddings = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.num_classes, use_bias=True)(embeddings)
  predictions = layers.Activation(activation=params.classifier_activation)(logits)
  return predictions, embeddings


def yamnet_frames_model(params):
  """Defines the YAMNet waveform-to-class-scores model.

  Args:
    params: An instance of Params containing hyperparameters.

  Returns:
    A model accepting (num_samples,) waveform input and emitting:
    - predictions: (num_patches, num_classes) matrix of class scores per time frame
    - embeddings: (num_patches, embedding size) matrix of embeddings per time frame
    - log_mel_spectrogram: (num_spectrogram_frames, num_mel_bins) spectrogram feature matrix
  """
  waveform = layers.Input(batch_shape=(None,), dtype=tf.float32)
  waveform_padded = pad_waveform(waveform, params)
  log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(
      waveform_padded, params)
  predictions, embeddings = yamnet(features, params)
  frames_model = Model(
      name='yamnet_frames', inputs=waveform,
      outputs=[predictions, embeddings, log_mel_spectrogram])
  return frames_model


def class_names(class_map_csv):
  """Read the class name definition file and return a list of strings."""
  if tf.is_tensor(class_map_csv):
    class_map_csv = class_map_csv.numpy()
  with open(class_map_csv) as csv_file:
    reader = csv.reader(csv_file)
    next(reader)   # Skip header
    return np.array([display_name for (_, _, display_name) in reader])


def yamnet_scream_detection(features, params):
    """Define the modified YAMNet model for scream detection.

    Only detects two classes: "scream" and "non_scream".

    Args:
      features: Input features, e.g., log mel spectrogram patches.
      params: An instance of Params containing hyperparameters.

    Returns:
      predictions: Output predictions for binary classification (scream vs. non_scream).
      embeddings: Embeddings per time frame.
    """
    net = layers.Reshape(
        (params.patch_frames, params.patch_bands, 1),
        input_shape=(params.patch_frames, params.patch_bands))(features)
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
    # Add more layers here for a smoother transition

    embeddings = layers.GlobalAveragePooling2D()(net)

    # Modify the final Dense layer to have 1 unit for binary classification
    predictions = layers.Dense(units=1, activation='sigmoid', use_bias=True)(embeddings)

    return predictions, embeddings



def create_modified_yamnet(num_classes=2):
    """Create a modified YAMNet model for detecting shouts.
    Args:
        num_classes (int): Number of output classes. Default is 2 for "shout" and "non-shout".

    Returns:
        tf.keras.Model: The modified YAMNet model for shout detection.
    """
    features = layers.Input(shape=(96, 64, 1), dtype=tf.float32)
    net = features
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)

    # Remove the GlobalAveragePooling2D and Dense layer for the original classes

    net = layers.Reshape((1024,))(net)
    # Add a new Dense layer for the modified classes (shout vs non-shout)
    logits = layers.Dense(units=num_classes, use_bias=True)(net)
    predictions = layers.Activation(activation='sigmoid')(logits)
    model = Model(name='yamnet_shout_detection', inputs=features, outputs=predictions)
    return model


def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
  """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""
  def _hann_window():
    return tf.reshape(
      tf.constant(
          (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
          ).astype(np.float32),
          name='hann_window'), [1, frame_length])
  

def waveform_to_log_mel_spectrogram_patches(waveform, params):
    """Compute log mel spectrogram patches of a 1-D waveform."""
    with tf.name_scope('log_mel_features'):
      # waveform has shape [<# samples>]

      # Convert waveform into spectrogram using a Short-Time Fourier Transform.
      # Note that tf.signal.stft() uses a periodic Hann window by default.
      window_length_samples = int(
        round(params.sample_rate * params.stft_window_seconds))
      hop_length_samples = int(
        round(params.sample_rate * params.stft_hop_seconds))
      fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
      num_spectrogram_bins = fft_length // 2 + 1
      if params.tflite_compatible:
        magnitude_spectrogram = _tflite_stft_magnitude(
            signal=waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length)
      else:
        magnitude_spectrogram = tf.abs(tf.signal.stft(
            signals=waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length))
      # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]

      # Convert spectrogram into log mel spectrogram.
      linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=params.mel_bands,
          num_spectrogram_bins=num_spectrogram_bins,
          sample_rate=params.sample_rate,
          lower_edge_hertz=params.mel_min_hz,
          upper_edge_hertz=params.mel_max_hz)
      mel_spectrogram = tf.matmul(
        magnitude_spectrogram, linear_to_mel_weight_matrix)
      log_mel_spectrogram = tf.math.log(mel_spectrogram + params.log_offset)
      # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

      # Frame spectrogram (shape [<# STFT frames>, params.mel_bands]) into patches
      # (the input examples). Only complete frames are emitted, so if there is
      # less than params.patch_window_seconds of waveform then nothing is emitted
      # (to avoid this, zero-pad before processing).
      spectrogram_hop_length_samples = int(
        round(params.sample_rate * params.stft_hop_seconds))
      spectrogram_sample_rate = params.sample_rate / spectrogram_hop_length_samples
      patch_window_length_samples = int(
        round(spectrogram_sample_rate * params.patch_window_seconds))
      patch_hop_length_samples = int(
        round(spectrogram_sample_rate * params.patch_hop_seconds))
      features = tf.signal.frame(
          signal=log_mel_spectrogram,
          frame_length=patch_window_length_samples,
          frame_step=patch_hop_length_samples,
          axis=0)
      # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]

      # Pad the features tensor to ensure it has a fixed number of patches (20 in this case).
      num_padding_patches = tf.maximum(0, 16 - tf.shape(features)[0])
      paddings = [[0, num_padding_patches], [0, 0], [0, 0]]
      features = tf.pad(features, paddings, "CONSTANT", constant_values=0)

      return log_mel_spectrogram, features
  

def pad_waveform(waveform, params):
    # Assuming waveform is 2D with shape (None, 32000), where 32000 is the waveform length
    waveform = waveform[:, 0]  # Take the first dimension to make it 1D
    waveform_len = tf.shape(waveform)[0]
    target_length = params.expected_waveform_length

    # Compute the number of zeros to add
    padding_length = target_length - waveform_len

    # Pad the waveform with zeros at the end
    paddings = tf.expand_dims([0, padding_length], axis=0)  # Reshape to 2D tensor
    waveform_padded = tf.pad(waveform, paddings)

    return waveform_padded



def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform


# waveform = wavfile.read('./AUDIOS/train/train_data/audio_grito.wav', )[1].astype(np.float32)
# wav_file_name = './AUDIOS/train/train_data/audio_grito.wav'
# sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
# sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# # Crear el modelo modificado para detección de gritos usando la función create_modified_yamnet.
# wav_data = wav_data / tf.int16.max
# waveform_padded = pad_waveform(waveform, params)
# log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(waveform_padded, params)

# # modified_model = create_modified_yamnet(2)
# model = yamnet(features, params)


def yamnet_scream_detection_frames_model(params):
    # Set the expected waveform length for padding
    # params.expected_waveform_length = 32000  # Replace this with your desired length

    waveform = layers.Input(batch_shape=(None, params.expected_waveform_length), dtype=tf.float32)
    waveform_padded = pad_waveform(waveform, params)
    log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(
        waveform_padded, params)
    predictions, embeddings = yamnet_scream_detection(features, params)
    frames_model = Model(
        name='yamnet_scream_detection_frames', inputs=waveform,
        outputs=[predictions, embeddings, log_mel_spectrogram])
    return frames_model


from tensorflow.keras.layers import Lambda

# Define la longitud máxima de las muestras de audio
max_audio_length = 80000
# Create the YAMNet scream detection model
# yamnet_scream_detection_model = yamnet_scream_detection_frames_model(params)

# def custom_pad(x):
#     return tf.compat.v1.pad(x, paddings=[[0, max_audio_length - tf.shape(x)[0]]], mode='CONSTANT', constant_values=0.0)

# # Utiliza la capa Lambda para aplicar la función personalizada de pad
# padded_input = Lambda(custom_pad)(yamnet_scream_detection_frames_model.input)  # Reemplaza 'input_layer' con la entrada real de tu modelo


# Compile the model
# yamnet_scream_detection_model.layers[-1].units = 1
# yamnet_scream_detection_model.layers[-1].activation = 'sigmoid'
# num_classes = 2  # Number of classes for binary classification
# yamnet_scream_detection_model.add(Dense(num_classes, activation='sigmoid'))

# yamnet_scream_detection_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])


# yamnet_scream_detection_model.summary()


# ENTRENAMIENTOOO

import os

non_scream_dir = './AUDIOS/train/train_data/non_scream'
scream_dir = './AUDIOS/train/train_data/scream'


def preprocess_audio_file(filepath, target_length=None):
    sample_rate, wav_data = wavfile.read(filepath)
    wav_data = np.mean(wav_data, axis=1).astype(np.float32)
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # print(wav_data.shape)
    
    #Pad or truncate the audio data to the target length
    if target_length is not None:
        if len(wav_data) < target_length:
            pad_length = target_length - len(wav_data)
            wav_data = np.pad(wav_data, (0, pad_length), mode='constant')
        elif len(wav_data) > target_length:
            wav_data = wav_data[:target_length]
    
    return wav_data


def load_data(data_dir, label, target_length=None):
    data = []
    labels = []
    target_length = 16000
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        
        wav_data = preprocess_audio_file(filepath, target_length)
        data.append(wav_data)
        labels.append(label)
    return data, labels

# Define the target length for padding/truncating the audio samples
target_length = 67368  # Use the maximum length among all samples

non_scream_data, non_scream_labels = load_data(non_scream_dir, 0)
scream_data, scream_labels = load_data(scream_dir, 1)

# Combina los datos y las etiquetas
all_data = non_scream_data + scream_data
all_labels = non_scream_labels + scream_labels

# # Convert the lists to TensorFlow tensors
# all_data = tf.convert_to_tensor(all_data)
# all_labels = tf.convert_to_tensor(all_labels)

# # Convierte las listas de Python a arrays de numpy
all_data = np.array(all_data)
all_labels = np.array(all_labels)

from sklearn.model_selection import train_test_split

# Divide los datos en conjuntos de entrenamiento y prueba
train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
# train_data = np.array(train_data)
# train_labels = np.array(train_labels)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels, num_classes=2)
train_labels_binary = np.argmax(train_labels, axis=1)
train_labels_binary = train_labels_binary.reshape(-1, 1)
print(train_data.shape)
print(train_labels_binary.shape)

# yamnet_scream_detection_model.fit(train_data, train_labels_binary, epochs=1, batch_size=32, validation_split=0.1)