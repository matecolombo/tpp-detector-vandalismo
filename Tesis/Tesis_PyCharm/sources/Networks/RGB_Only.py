def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import os
from time import time
import cv2
import sys


import tensorflow as tf
import tensorflow.keras as keras
#from keras.utils import Sequence
from tensorflow.keras import utils  
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model  # Sequential
from tensorflow.keras.layers import Input
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply
from keras.regularizers import l2
from tensorflow.keras.models import save_model

from tensorflow.keras.mixed_precision import global_policy, set_global_policy, Policy
import tensorflow_model_optimization as tfmot
import keras.backend as K
from keras.callbacks import LearningRateScheduler

import tensorflow.keras.backend as backend
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger  # ModelCheckpoint,


# Definir el tipo de precisión mixta deseada (puede ser 'float16' o 'bfloat16')
policy = global_policy()
if policy.name == 'float32':
    policy = Policy('mixed_float16')
set_global_policy(policy)


#os.environ['TF_GPU_ALLOCATOR_MAX_ALLOC_PERCENT'] = '100'

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



class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args: 
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        #print("__init__")
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data() 
        # Print basic statistics information
        self.print_stats()
        return None
        
    def search_data(self):
        #print("search_data")
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = utils.to_categorical(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict
    
    def print_stats(self):
        #print("print_stats")
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None
    
    def __len__(self):
        #print("__len__")
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        #print("__getitem__")
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        #print("on_epoch_end")
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        #print("data_generation")
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
      
    def normalize(self, data):
        #print("normalize")
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std
    
    def random_flip(self, video, prob):
        #print("random_flip")
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    
    
    def uniform_sampling(self, video, target_frames=64):
        #print("uniform_sampling")
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        if num_pad>0:
            padding = [video[i] for i in range(-num_pad,0)]
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def dynamic_crop(self, video):
        #print("dynamic_crop")
        # extract layer of optical flow from video
        opt_flows = video[...,3]
        # sum of optical flow magnitude of individual frame
        magnitude = np.sum(opt_flows, axis=0)
        # filter slight noise by threshold 
        thresh = np.mean(magnitude)
        magnitude[magnitude<thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # randomly choose some candidates for x and y 
        x_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(224), size=10, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        # avoid to beyond boundaries of array
        x = max(56,min(x,167))
        y = max(56,min(y,167))
        # get cropped video 
        return video[:,x-56:x+56,y-56:y+56,:]  
    
    def color_jitter(self,video):
        #print("color_jitter")
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video
    
    def load_data(self, path):
       # print("load_data")
        try:
            # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
            data = np.load(path, mmap_mode='r', allow_pickle=True)
            data = data.astype(np.float32)

            # sampling 64 frames uniformly from the entire video
            data = self.uniform_sampling(video=data, target_frames=64)

            # whether to utilize the data augmentation
            if self.data_aug:
                data[..., :3] = self.color_jitter(data[..., :3])
                data = self.random_flip(data, prob=0.5)

            # normalize rgb images and optical flows, respectively
            data[..., :3] = self.normalize(data[..., :3])
            data = data[..., :3]
            #print(data.shape)
            return data
    
        except Exception as e:
            # If there's an error loading the file, print an error message and return None
            print(f"Error loading file: {path}. Error message: {e}")
            return None
   
    
inputs = Input(shape=(64,224,224,3))

#####################################################
rgb = inputs
rgb = Conv3D(
    16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = Conv3D(
    16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

rgb = Conv3D(
    16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = Conv3D(
    16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

rgb = Conv3D(
    32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = Conv3D(
    32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

rgb = Conv3D(
    32, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = Conv3D(
    32, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)

#####################################################
x = MaxPooling3D(pool_size=(8,1,1))(rgb)

#####################################################
x = Conv3D(
    64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2,2,2))(x)

x = Conv3D(
    64, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2,2,2))(x)

x = Conv3D(
    128, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    128, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2,3,3))(x)

#####################################################
x = Flatten()(x)
x = Dense(128,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
pred = Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=pred)

model.summary()


def scheduler(epoch):
    # Every 10 epochs, the learning rate is reduced to 1/10 of the original
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)




# Saving the best model and training logs

class MyCbk(keras.callbacks.Callback):
    def __init__(self, model_to_save):
        super().__init__()
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('Logs/model_at_epoch_%d.h5' % (epoch + 1))


check_point = MyCbk(model)

filename = 'Logs/ours_log.csv'
csv_logger = CSVLogger(filename, separator=',', append=True)

callbacks_list = [check_point, csv_logger, reduce_lr]

#callbacks_list = [reduce_lr]

adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 30
num_workers = 16
batch_size = 2#8

dataset = 'RWF2000-opt'
directory_train = '../../Dataset/Numpy_Images/train'
directory_val = '../../Dataset/Numpy_Images/val'

print("Train data generator")

train_generator = DataGenerator(directory=directory_train.format(dataset), ###############
                                batch_size=batch_size,
                                data_augmentation= True)

print("Validation data generator")

val_generator = DataGenerator(directory=directory_val.format(dataset),   ###############
                              batch_size=batch_size,
                              data_augmentation=False)

'''


if np.any(np.isnan(train_generator)) or np.any(np.isnan(val_generator)):
    # Handle missing values (e.g., replace them with appropriate values or remove them)
    train_generator = np.nan_to_num(train_generator)
    val_generator = np.nan_to_num(val_generator)

print("Empieza el entrenamiento")
hist = model.fit(
    x=train_generator, 
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1, 
    epochs=num_epochs,
    workers=num_workers ,
    max_queue_size=8,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator))


keras_file = 'RGB_model.h5'
save_model(model, keras_file)
#save_model(model_for_pruning, keras_file)

'''
initial_sparsity = 0.0
final_sparsity = 0.5
begin_step = 2000
end_step=4000
power=3  
frequency=100

# Temporarily set the global default floating-point precision to 'float32'
previous_floatx = tf.keras.backend.floatx()
tf.keras.backend.set_floatx('float32')

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=initial_sparsity,
    final_sparsity=final_sparsity,
    begin_step=begin_step,
    end_step=end_step,
#   power=power,  
#   frequency=frequency
)

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

# Add the UpdatePruningStep callback
callbacks_list = [check_point, csv_logger, reduce_lr, tfmot.sparsity.keras.UpdatePruningStep()]

# Revert the global default floating-point precision to its original setting
tf.keras.backend.set_floatx(previous_floatx)

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model_for_pruning.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# start to train
#hist = model.fit(
hist = model_for_pruning.fit(
    x=train_generator,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1,
    epochs=num_epochs,
    workers=num_workers,
    max_queue_size=4,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)


keras_file = 'Ro_model.h5'
#keras.models.save_model(model, keras_file)
keras.models.save_model(model_for_pruning, keras_file)




