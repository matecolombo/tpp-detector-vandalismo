import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.lite.python.interpreter import OpResolverType
import tensorflow_model_optimization as tfmot
import tempfile


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


def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)


################### Create TFLite file ###################

# Modo Normal

def model_to_TFlite(model, tflite_file):
    # Convert the Keras file to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)


# Modo Reduced Float

def model_to_TFlite_Reduced_Float(model, tflite_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)


# Hybrid Quantization
def model_to_TFlite_HybridQuantization(model, tflite_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)


# Integer Quantization
def model_to_TFlite_IntegerQuantization(model, tflite_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = input_generator
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)


# Full Integer Quantization (Only Integers)
# https://www.tensorflow.org/lite/performance/post_training_quantization
def model_to_TFlite_FullIntegerQuantization(model, tflite_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = input_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)


################### Pruning ###################
# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras


batch_size = 128
epochs = 2
validation_split = 0.1  # 10% of training set will be used for validation set.

initial_sparsity = 0.0
final_sparsity = 0.5
begin_step = 2000
end_step=4000
power=3  
frequency=100


def pruning(model, input_output, model_pruned_file):
    batch_size = len(input_output)
        
    # Train the digit classification model

    '''
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


    _, baseline_model_accuracy = model.evaluate(
        input_output, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    '''

    # num_images = input_output.shape[0] * (1 - validation_split)
    # end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs


    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
    #   power=power,  
    #   frequency=frequency
    )

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    model_for_pruning.compile(optimizer='adam',
                            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            loss='mean_squared_error',
                            metrics=['accuracy'])

    logdir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]


    model_for_pruning.fit(input_output,
                        batch_size=batch_size, epochs=epochs,  # validation_split=validation_split,
                        callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(train_images, verbose=0)
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    open(model_pruned_file, "wb").write(model_for_pruning)
    '''
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    _, pruning_file = tempfile.mkstemp('.h5')
    

    tf.keras.models.save_model(model_for_export, pruning_file, include_optimizer=False)
    
    print('Saved pruned Keras model to:', pruning_file)
    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    
    model_to_TFlite(model_for_pruning, tflite_file_pruning)
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_file_pruning)))
    
    model_to_TFlite_Reduced_Float(model_for_pruning, tflite_file_pruning_ReducedFloat)
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_file_pruning_ReducedFloat)))
    
    model_to_TFlite_HybridQuantization(model_for_pruning, tflite_file_pruning_HybridQuantization)
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_file_pruning_HybridQuantization)))
    
    model_to_TFlite_IntegerQuantization(model_for_pruning, tflite_file_pruning_IntegerQuantization)
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_file_pruning_IntegerQuantization)))
    
    model_to_TFlite_FullIntegerQuantization(model_for_pruning, tflite_file_pruning_FullIntegerQuantization)
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_file_pruning_FullIntegerQuantization)))

    '''


################### Prediction with TFLite file ###################
def predict_TFlite(tflite_file, input):
    interpreter = tf.lite.Interpreter(tflite_file)

    input_shape = interpreter.get_input_details()[0]['shape']

    # interpreter.resize_tensor_input(0, input_shape, strict=False)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(input_details)

    for i in range(0, len(input)):
        input = np.array(input[i], dtype=np.float32)
        interpreter.set_tensor(input_details['index'], input)

        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details['index'])
        print(i, predictions[0][0])