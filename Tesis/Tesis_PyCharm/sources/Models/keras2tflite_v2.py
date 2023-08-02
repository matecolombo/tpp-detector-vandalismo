import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import os


def convert_bytes(size, unit=None):
    print('hola')
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

def get_file_size(file_path):

    size = os.path.getsize(file_path)
    return size

model_file = "./keras_model.h5"
model_dir = ""
model = load_model(model_file, compile=False)
TF_LITE_MODEL_FILE_NAME = "tf_lite_model_key.tflite"


saved_model_dir = model_dir
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
print(converter)
tflite_model = converter.convert()

fo = open(TF_LITE_MODEL_FILE_NAME, "wb")
convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), 'KB')
fo.write(tflite_model)
fo.close()

# convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), 'KB')
