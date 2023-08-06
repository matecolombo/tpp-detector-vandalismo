import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import sys
import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.models import save_model
from tensorflow.keras.mixed_precision import global_policy, set_global_policy, Policy
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input
from Data_Generator import DataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import CSVLogger  # ModelCheckpoint,
from tensorflow.keras.models import load_model
from Create_OPT_Model import create_son_model, create_model,custom_loss
from GPU_Config import gpu_config

gpu_config()

# Definir el tipo de precisión mixta deseada (puede ser 'float16' o 'bfloat16')
policy = global_policy()
if policy.name == 'float32':
    policy = Policy('mixed_float16')
set_global_policy(policy)

# Supongamos que tienes los datos de entrada para el modelo

inputs = Input(shape=(64, 224, 224, 2))
# Crea el modelo padre (modelo original)
#model_file = '../../Models/keras_model.h5' #v'OPT_model.h5' #
#modelo_padre = load_model(model_file, compile=False)
modelo_padre = load_model('./modelo_padre')
modelo_padre.summary()
#modelo_padre.save('./modelo_padre', save_format='tf')

# Entrena el modelo padre con tus datos (omitiendo el proceso de entrenamiento ya que no es relevante para la transferencia de pesos)
# Crea el modelo hijo
model = create_son_model(inputs)

# Supongamos que tienes el modelo hijo definido en otra función
# Si las capas semejantes están en el rango de 12 a 23, puedes usar la siguiente lista de índices
capas_semejantes = list(range(0,28))#list(range(12, 24)) #[4,6,8,10,12,14,16,18,20,22,24,26] # Índices de las capas semejantes en el modelo padre #

# Supongamos que tienes el modelo hijo definido en otra función
# Si las capas semejantes están en el rango de 12 a 23, puedes usar la siguiente lista de índices
capas_semejantes = [index for index in capas_semejantes if
                    isinstance(modelo_padre.get_layer(index=index), keras.layers.Conv3D)]

# Extrae los pesos de las capas semejantes del modelo padre
pesos_capas_semejantes = [modelo_padre.get_layer(index=index).get_weights() for index in capas_semejantes]

# Transfiere los pesos de las capas semejantes del modelo padre al modelo hijo
for i, index in enumerate(capas_semejantes):
    layer = model.get_layer(index=index)
    if layer.weights:
        padre_weights_shape = pesos_capas_semejantes[i][0].shape
        hijo_weights_shape = layer.get_weights()[0].shape

        if padre_weights_shape == hijo_weights_shape:
            layer.set_weights(pesos_capas_semejantes[i])
            layer.trainable = False
        else:
            print(f"Los pesos para la capa {index} no son compatibles y no se transferirán.")
    else:
        print(f"La capa {index} no tiene pesos y no se transferirán.")

# Continúa con el entrenamiento del modelo hijo utilizando los datos específicos para esta tarea
# Compila y entrena el modelo hijo con tus datos específicos para la tarea

def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = tensorflow.keras.backend.get_value(model.optimizer.lr)
        tensorflow.keras.backend.set_value(model.optimizer.lr, lr * 0.5)
    return tensorflow.keras.backend.get_value(model.optimizer.lr)


class MyCbk(keras.callbacks.Callback):
    def _init_(self, model_to_save):
        super()._init_()
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('Logs/model_at_epoch_%d.h5' % (epoch + 1))


# Definir el tipo de precisión mixta deseada (puede ser 'float16' o 'bfloat16')
policy = global_policy()
if policy.name == 'float32':
    policy = Policy('mixed_float16')
set_global_policy(policy)

sgd = SGD(learning_rate=0.01, momentum=0.9)

model.compile(optimizer=sgd, loss=custom_loss, metrics=['accuracy'])


#############################################

reduce_lr = LearningRateScheduler(scheduler)
check_point = MyCbk(model)
filename = 'Logs/ours_log.csv'
csv_logger = CSVLogger(filename, separator=',', append=True)
# callbacks_list = [reduce_lr]
callbacks_list = [check_point, csv_logger, reduce_lr]
directory_train = '../../../Dataset/Numpy_Images/train'
directory_val = '../../../Dataset/Numpy_Images/val'
num_epochs = 30
num_workers = 16
batch_size = 2  # 16
dataset = 'RWF2000-opt'

train_generator = DataGenerator(directory=directory_train.format(dataset),
                                batch_size=batch_size,
                                data_augmentation=True)

val_generator = DataGenerator(directory=directory_val.format(dataset),
                              batch_size=batch_size,
                              data_augmentation=False)

hist = model.fit(
    x=train_generator,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1,
    epochs=num_epochs,
    workers=num_workers,
    max_queue_size=8,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator))

keras_file = 'OPT_model.h5'
save_model(model, keras_file)