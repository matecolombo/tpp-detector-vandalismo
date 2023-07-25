import sys
import os
import shutil
from time import time
import tensorflow as tf
import numpy as np
from memory_profiler import profile
from memory_profiler import LogFile
from tensorflow.lite.python.interpreter import OpResolverType

# sys.stdout = LogFile('memory_profile_log')

sys.path.append('../Preprocess')
sys.path.append('../Networks')
sys.path.append('../Models')

from Preprocess_webcam import get_camera_fps, Save2Npy, getVideo
from Network_Functions import DataGenerator_tflite

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

video_dir = '../Preprocess/Video_Webcam/AVI'
npy_dir = '../Preprocess/Video_Webcam/NPY'
video_dir_2 = 'Video_Webcam/AVI'
npy_dir_2 = 'Video_Webcam/NPY'
discard_dir = '../Preprocess/Video_Webcam/Discard'
prediction_dir = '../Preprocess/Video_Webcam/Predictions'
dataset = 'ViolentFlow-opt'
tflite_model_file = "../Models/tf_lite_model_v2.tflite"
loss = 'categorical_crossentropy'

batch_size = 2
learning_rate = 0.01
decay = 1e-6
momentum = 0.9
num_videos = 5

# fps = get_camera_fps()

# model_size = os.path.getsize(tflite_model_file) / 1048576
# print("Tamaño del modelo: ", model_size, " MB")
# Cargar modelo TFLite y allocate tensores
interpreter = tf.lite.Interpreter(model_path=tflite_model_file,
                                  model_content=None,
                                  experimental_delegates=None,
                                  num_threads=1,
                                  experimental_op_resolver_type=OpResolverType.AUTO,
                                  experimental_preserve_all_tensors=False)
print(interpreter)
'''


def check_no_numpy_array():
    # Obtén una lista de todas las variables globales en el espacio de nombres actual
    global_vars = list(globals().values())

    # Recorre todas las variables globales
    for var in global_vars:
        # Verifica si la variable es una instancia de NumPy ndarray
        if isinstance(var, np.ndarray):
            # Verifica si la matriz de NumPy apunta a un buffer interno
            if np.may_share_memory(var, np.zeros(1)):
                # Si se encuentra una matriz de NumPy que apunte a un buffer interno, lanza una excepción
                raise RuntimeError("Existen objetos de NumPy que apuntan a memoria interna activa.")
    print("No hay objetos de NumPy que apuntan a memoria interna activa.")

check_no_numpy_array()
'''

# Obtener tensores de entrada y salida
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# input_shape = input_details['shape']
# print(input_shape)

'''
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
if not os.path.exists(npy_dir):
    os.makedirs(npy_dir)
if not os.path.exists(discard_dir):
    os.makedirs(discard_dir)
if not os.path.exists(discard_dir):
    os.makedirs(discard_dir)
'''

# for i in range(1, num_videos + 1):

# Take Video
# video_path = getVideo(video_dir, fps)

# Take Video and adapt it to npy
# Save2Npy(video_dir, npy_dir)

# Generate input
input_model = DataGenerator_tflite(directory=npy_dir_2.format(dataset),
                                   batch_size_data=batch_size,
                                   data_augmentation=False)
batch_x, batch_y = input_model.__getitem__(0)  # Use index 0 to get the first batch

interpreter.set_tensor(input_details['index'], batch_x)

# Predict violence
# time_before = time()

'''
@profile
def invocar():
    interpreter.invoke()


invocar()
'''
interpreter.invoke()
# delta_time = time() - time_before
# print("Tiempo de predicción: ", delta_time)

predictions = interpreter.get_tensor(output_details['index'])

print(predictions)
'''

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(tflite_model_file)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'])




# Graph
# print_prediction(predictions)

# Move all the npy files to the discard directory
for file in os.listdir(npy_dir):
    file_path = os.path.join(npy_dir, file)
    shutil.move(file_path, discard_dir)

for file in os.listdir(video_dir):
    file_path = os.path.join(video_dir, file)
    shutil.move(file_path, discard_dir)
'''
