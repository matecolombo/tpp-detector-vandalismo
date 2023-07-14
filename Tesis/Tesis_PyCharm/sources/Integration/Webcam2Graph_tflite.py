import sys
import os
import shutil
from time import time
import tensorflow as tf
import numpy as np

sys.path.append('../Preprocess')
sys.path.append('../Networks')
sys.path.append('../Models')

from Preprocess_webcam import get_camera_fps, Save2Npy, getVideo
from Network_Functions import DataGenerator_tflite
from Print_prediction import print_prediction

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

video_dir = '../Preprocess/Video_Webcam/AVI'
npy_dir = '../Preprocess/Video_Webcam/NPY'
discard_dir = '../Preprocess/Video_Webcam/Discard'
prediction_dir = '../Preprocess/Video_Webcam/Predictions'
dataset = 'ViolentFlow-opt'
tflite_model_file = "../Models/tf_lite_model.tflite"
loss = 'categorical_crossentropy'

batch_size = 2
learning_rate = 0.01
decay = 1e-6
momentum = 0.9
num_videos = 5
fps = get_camera_fps()

model_size = os.path.getsize(tflite_model_file) / 1048576
print("Tamaño del modelo: ", model_size, " MB")

# Cargar modelo TFLite y alocar tensores
interpreter = tf.lite.Interpreter(tflite_model_file)
interpreter.allocate_tensors()

# Obtener tensores de entrada y salida
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_shape = input_details['shape']
print(input_shape)

if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)
if not os.path.exists(discard_dir):
    os.makedirs(discard_dir)
if not os.path.exists(discard_dir):
    os.makedirs(discard_dir)

for i in range(1, num_videos + 1):

    # Take Video
    #  video_path = getVideo(video_dir, fps)

    # Take Video and adapt it to npy
    #  Save2Npy(video_dir, npy_dir)

    # Generate input
    input_model = DataGenerator_tflite(directory=npy_dir.format(dataset),
                                       batch_size_data=batch_size,
                                       data_augmentation=False)

    interpreter.set_tensor(input_details['index'], input_model)

    # Predict violence
    # time_before = time()
    interpreter.invoke()
    # delta_time = time() - time_before
    # print("Tiempo de predicción: ", delta_time)

    print(interpreter.get_tensor(output_details['index']).shape)

    predictions = interpreter.get_tensor(output_details['index'])

    # Graph
    print_prediction(predictions)

    # Move all the npy files to the discard directory
    for file in os.listdir(npy_dir):
        file_path = os.path.join(npy_dir, file)
        shutil.move(file_path, discard_dir)
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        shutil.move(file_path, discard_dir)
