import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import OpResolverType

keras_file = "linear.h5"
tflite_file = "linear.tflite"
# Train a very linear model
'''
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Dataset
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 0.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

# Test a simple prediction
print(model.predict([10.0]))

keras.models.save_model(model, keras_file)
'''

# Create TFLite file
'''
model = load_model(keras_file, compile=False)
# Convert the Keras file to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_file,"wb").write(tflite_model)
'''


interpreter = tf.lite.Interpreter(tflite_file)

input_shape = interpreter.get_input_details()[0]['shape']

# interpreter.resize_tensor_input(0, input_shape, strict=False)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)

for i in range(0, 100):
    input = np.array([[i]], dtype=np.float32)
    interpreter.set_tensor(input_details['index'], input)

    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details['index'])
    print(i, predictions[0][0])
