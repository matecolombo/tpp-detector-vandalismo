import sys

sys.path.append('../Models')
sys.path.append('../Networks')
sys.path.append('../Preprocess')

from Network_Functions import DataGenerator_adapted
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

batch_size = 2
dataset = 'ViolentFlow-opt'

model_file = "../Models/keras_model.h5"
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#model = load_model('keras_model.h5', compile=False)

model = load_model(model_file, compile=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

discard_directory = '../../Test/Npy/Discard'

val_generator = DataGenerator_adapted(directory='../../Test/Npy'.format(dataset),
                                      #discard_directory=discard_directory,
                                      batch_size_data=batch_size,
                                      data_augmentation=False)

predictions = model.predict(val_generator)  # ,allow_pickle=True)

print(predictions)
