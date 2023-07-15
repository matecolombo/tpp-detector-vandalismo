import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

model_file = "keras_model.h5"
model_dir = ""
model = load_model(model_file, compile=False)
TF_LITE_MODEL_FILE_NAME = "tf_lite_model_key.tflite"


saved_model_dir = model_dir
converter = tf.lite.TFLiteConverter.from_saved_model(model_file, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

fo = open(TF_LITE_MODEL_FILE_NAME, "wb")
fo.write(tflite_model)
fo.close()