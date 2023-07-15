import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
TFLITE_MODEL_FILE_V1 = "tf_lite_model_v2.tflite"
TFLITE_MODEL_FILE_V2 = 'tf_lite_model_v2.tflite'

model_file = TFLITE_MODEL_FILE_V1
interprete = tf.lite.Interpreter(model_file)

input_shape = interprete.get_input_details()[0]['shape']

print(input_shape)
interprete.resize_tensor_input(0, input_shape, strict=True)
interprete.allocate_tensors()

