import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import OpResolverType

keras_file = "resnet50_edited.h5"
tflite_file = "resnet50_edited.tflite"

# Create TFLite file

model = load_model(keras_file, compile=False)
# Convert the Keras file to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_file, "wb").write(tflite_model)
