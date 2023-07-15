import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

model_file = "./keras_model.h5"
model_dir = "./"
model = load_model(model_file, compile=False)
TF_LITE_MODEL_FILE_NAME = "tf_lite_model_v2.tflite"


converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
'''
converter.optimizations = [tf.lite.Optimize.DEFAULT]
'''
tflite_model = converter.convert()


tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name,"wb").write(tflite_model)

#convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), "KB" )

'''
converter = tf.lite.TFLiteConverter.from_saved_model(model_file)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
'''