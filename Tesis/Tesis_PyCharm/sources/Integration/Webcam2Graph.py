import sys
import os
import shutil
from time import time
sys.path.append('../Preprocess')
sys.path.append('../Networks')
sys.path.append('../Models')

from Preprocess_webcam import get_camera_fps, Save2Npy, getVideo
from Network_Functions import DataGenerator_adapted
from Print_prediction import print_prediction

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

video_dir = '../Preprocess/Video_Webcam/AVI'
npy_dir = '../Preprocess/Video_Webcam/NPY'
discard_dir = '../Preprocess/Video_Webcam/Discard'
prediction_dir = '../Preprocess/Video_Webcam/Predictions'
dataset = 'ViolentFlow-opt'
model_file = "../Models/keras_model.h5"
tflite_model_file = "../Models/tf_lite_model.tflite"
loss = 'categorical_crossentropy'

batch_size = 2
learning_rate = 0.01
decay = 1e-6
momentum = 0.9
num_videos = 5
fps = get_camera_fps()

model_size = os.path.getsize(model_file)/1048576
print("Tamaño del modelo de keras: ", model_size, " MB")

model_size = os.path.getsize(tflite_model_file)/1048576
print("Tamaño del modelo: ", model_size, " MB")

model = load_model(model_file, compile=False)
#tflite_model = load_model(tflite_model_file, compile=False)

'''
sgd = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=True)
model.compile(optimizer=sgd,
              loss=loss,
              metrics=['accuracy'])
'''


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
    video_path = getVideo(video_dir, fps)

    # Take Video and adapt it to npy
    Save2Npy(video_dir, npy_dir)

    # Generate input
    input_model = DataGenerator_adapted(directory=npy_dir.format(dataset),
                                        batch_size_data=batch_size,
                                        data_augmentation=False)

    time_before = time()
    # Predict violence
    predictions = model.predict(input_model)
    delta_time = time() - time_before
    print("Tiempo de predicción: ", delta_time)

    # Graph
    print_prediction(predictions)

    # Move all the npy files to the discard directory
    for file in os.listdir(npy_dir):
        file_path = os.path.join(npy_dir, file)
        shutil.move(file_path, discard_dir)
    for file in os.listdir(video_dir):
        file_path = os.path.join(video_dir, file)
        shutil.move(file_path, discard_dir)
