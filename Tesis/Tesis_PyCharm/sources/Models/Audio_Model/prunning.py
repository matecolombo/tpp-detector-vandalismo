
def pruning_post_training(model, model_dir, pruned_file):
    # model.load_weights(f'{model_dir}/pesos.h5')
    directory_val = '../../Dataset/Numpy_Images/val'
    num_epochs = 30
    num_workers = 16
    batch_size = 2  # 16
    dataset = 'RWF2000-opt'
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    # Cargar los pesos previamente entrenados
    # weight_file = f"{model_dir}/pesos.h5"

    # model.load_weights(weight_file)

    val_generator = DataGenerator(directory=directory_val.format(dataset),
                                  batch_size=batch_size,
                                  data_augmentation=False)

    end_step = 4000
    # Define la proporción de poda deseada (por ejemplo, 0.5 para eliminar el 50% de las conexiones menos importantes)
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.5,
                                                                 begin_step=0,
                                                                 end_step=end_step),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }

    # Aplica la poda a las capas del modelo
    pruned_model_file = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Compila el modelo podado
    pruned_model_file.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # Entrena el modelo podado con los mismos datos de entrenamiento que utilizaste anteriormente
    pruned_model_file.fit(val_generator, epochs=num_epochs, validation_data=val_generator)
    # Guarda el modelo podado
    pruned_model_file.save_weights(pruned_file)

    return pruned_model_file



import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn


from tensorflow.keras.layers import Input
from Data_Generator import DataGenerator
from tensorflow.keras.models import load_model
from GPU_Config import gpu_config
import tensorflow_model_optimization as tfmot
from tensorflow.keras.optimizers import SGD
from Create_OPT_Model import custom_loss

gpu_config()

directory_train = '../../../Dataset/Numpy_Images/train'
directory_val = '../../../Dataset/Numpy_Images/val'
num_epochs = 30
num_workers = 16
batch_size = 2  # 16
dataset = 'RWF2000-opt'


inputs = Input(shape=(1405, 129, 3))

model = load_model('.MovileNetV2_adapt.h5')
model.summary()


sgd = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss=custom_loss, metrics=['accuracy'])

train_generator = DataGenerator(directory=directory_train.format(dataset),
                                batch_size=batch_size,
                                data_augmentation=True)

val_generator = DataGenerator(directory=directory_val.format(dataset),
                              batch_size=batch_size,
                              data_augmentation=False)

# num_images = train_images.shape[0] * (1 - validation_split)
# end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Crear una función de callback de poda

pruning_params = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.5, # 0.0,
    final_sparsity=0.8, # 0.5,
    begin_step=0, #2000,
    end_step=4000,  # end_step,
 #   power=3,  ######
 #   frequency=100  #####
)

pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

# Aplicar la poda al modelo
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Entrenar el modelo con la poda
# model_for_pruning.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[pruning_callback])

# hist = model.fit(
#     x=train_generator,
#     validation_data=val_generator,
#     callbacks=[pruning_callback],
#     verbose=1,
#     epochs=num_epochs,
#     workers=num_workers,
#     max_queue_size=8,
#     steps_per_epoch=len(train_generator),
#     validation_steps=len(val_generator))

# Después de la poda, es importante 'finalizar' el modelo para aplicar realmente la poda a los parámetros
model_for_pruning = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# Guardar el modelo podado
model_for_pruning.save('MovileNetV2_adapt_pruned.h5')