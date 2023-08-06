import os
import tensorflow as tf


def gpu_config():
    # os.environ['TF_GPU_ALLOCATOR_MAX_ALLOC_PERCENT'] = '100'

    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    YOUR_MEMORY_LIMIT_IN_BYTES = 7000000000

    # Configurar las opciones de la GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Restrict TensorFlow to only allocate a specific amount of GPU memory
             tf.config.experimental.set_virtual_device_configuration(gpus[0], [
               tf.config.experimental.VirtualDeviceConfiguration(memory_limit=YOUR_MEMORY_LIMIT_IN_BYTES)])
             #tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)