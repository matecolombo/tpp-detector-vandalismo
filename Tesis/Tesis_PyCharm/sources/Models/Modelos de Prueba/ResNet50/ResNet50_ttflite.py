
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.lite.python.interpreter import OpResolverType
import cv2

keras_file = "resnet50_edited.h5"
tflite_file = "resnet50_edited.tflite"

roses = list(data_dir.glob('roses/*'))
print(roses[0])
PIL.Image.open(str(roses[0]))



interpreter = tf.lite.Interpreter(tflite_file)

input_shape = interpreter.get_input_details()[0]['shape']

# interpreter.resize_tensor_input(0, input_shape, strict=False)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)

for i in range(0, 100):
    image = cv2.imread(str(roses[i]))
    image_resized = cv2.resize(image, (img_height, img_width))
    input = np.expand_dims(image_resized, axis=0)
    interpreter.set_tensor(input_details['index'], input)

    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details['index'])
    output_class = class_names[np.argmax(predictions[0][0])]
    print(i, "The predicted class is", output_class)
