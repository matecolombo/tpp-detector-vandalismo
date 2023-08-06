import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import tempfile
import os
from Network_Functions import DataGenerator_adapted


model_file = "./keras_model.h5"
model_dir = "./"
model = load_model(model_file, compile=False)
TF_LITE_MODEL_FILE_NAME = "tf_lite_model_int_quant_v2.tflite"


def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)


def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

def get_file_size(file_path):

    size = os.path.getsize(file_path)
    return size

input_data = "../Preprocess/Video_Webcam/INPUT_DATA"

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

_, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)
print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))


# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]

# reduce float 
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

# integer quantization
# calibration_steps = 100
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# def generate_numpy_data(directory):
#     # Initialize the DataGenerator_adapted with batch_size_data=1 and data_augmentation=False
#     data_generator = DataGenerator_adapted(directory, batch_size_data=1, shuffle=False, data_augmentation=False)
    
#     # Create an empty list to store the processed numpy data
#     numpy_data_list = []
    
#     # Loop over the data generator to obtain 100 batches of data
#     for batch_idx in range(100):
#         batch_x, _ = data_generator._getitem_(batch_idx)
#         # Since batch_x is a list with a single element (data for the batch), we extract that element
#         batch_x = batch_x[0]
#         numpy_data_list.append(batch_x)
    
#     # Return the list of processed numpy data
#     return numpy_data_list

# def data_generator():
#     for i in range(calibration_steps):
#         yield [input_data]

# converter.respresentative_dataset = data_generator


tflite_model = converter.convert()
tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name,"wb").write(tflite_model)

convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), 'KB')



'''
converter = tf.lite.TFLiteConverter.from_saved_model(model_file)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
'''