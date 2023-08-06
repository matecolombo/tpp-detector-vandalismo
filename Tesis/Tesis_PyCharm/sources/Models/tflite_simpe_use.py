import sys
import os

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

from tensorflow.python.compiler.tensorrt import trt_convert as trt


sys.path.append('../Models')
sys.path.append('../Networks')
sys.path.append('../Preprocess')

from Network_Functions import DataGenerator_tflite


TFLITE_MODEL_FILE_V1 = "tf_lite_model.tflite"
TFLITE_MODEL_FILE_V2 = 'tf_lite_model_v2.tflite'
npy_dir = "../Preprocess/Video_Webcam/NPY"
model_file = TFLITE_MODEL_FILE_V1
interprete = tf.lite.Interpreter(model_file)

input_shape = interprete.get_input_details()[0]['shape']

print(input_shape)
interprete.resize_tensor_input(0, input_shape, strict=True)
interprete.allocate_tensors()

input_details = interprete.get_input_details()[0]
output_details = interprete.get_output_details()[0]

batch_size = 2
dataset = 'ViolentFlow-opt'

# Generate input
input_model = DataGenerator_tflite(directory=npy_dir.format(dataset),
                                   batch_size_data=batch_size,
                                   data_augmentation=False)
batch_x, batch_y = input_model.__getitem__(0)  # Use index 0 to get the first batch

interprete.set_tensor(input_details['index'], batch_x)

interprete.invoke()
# delta_time = time() - time_before
# print("Tiempo de predicción: ", delta_time)

predictions = interprete.get_tensor(output_details['index'])

print(predictions)

