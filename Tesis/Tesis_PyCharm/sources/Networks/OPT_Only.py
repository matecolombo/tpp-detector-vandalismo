import numpy as np
import os
from time import time
import cv2
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import save_model

from tensorflow.keras.mixed_precision import global_policy, set_global_policy, Policy
#import tensorflow_model_optimization as tfmot
#import keras.backend as K
#from keras.callbacks import LearningRateScheduler

# Definir el tipo de precisión mixta deseada (puede ser 'float16' o 'bfloat16')
policy = global_policy()
if policy.name == 'float32':
    policy = Policy('mixed_float16')
set_global_policy(policy)

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
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = utils.to_categorical(range(len(self.dirs)))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # append the each file path, and keep its label
                X_path.append(file_path)
                Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
        for i, label in enumerate(self.dirs):
            print('%10s : ' % (label), i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

   
    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y
    '''

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)

        if batch_x is None or batch_y is None:
            # Return None for both batch_x and batch_y if the batch contains only invalid data
            return None, None

        return batch_x, batch_y
    '''

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    '''
    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
    

    def data_generation(self, batch_path):
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]

        batch_x = np.stack(batch_x)  # Use np.stack to create consistent shape
        batch_y = np.array(batch_y)

        return batch_x, batch_y
    '''


    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(d) for d in batch_path]
        batch_y = [self.Y_dict[b] for b in batch_path]

        # transfer the data format and take one-hot coding for labels
       # batch_x = np.array(batch_x)
       # batch_y = np.array(batch_y)
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        # Stack batch_x and batch_y into numpy arrays
       # batch_x = np.stack(batch_x, axis=0)
       # batch_y = np.stack(batch_y, axis=0)

        # Normalize batch_x
        batch_x = self.normalize(batch_x)

        return batch_x, batch_y



    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    '''


    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        if np.isnan(mean) or np.isnan(std) or std == 0:
            return data
        return (data - mean) / std

    '''

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video

    def uniform_sampling(self, video, target_frames=64):
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        if num_pad > 0:
            padding = [video[i] for i in range(-num_pad, 0)]
            sampled_video += padding
            # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    def dynamic_crop(self, video):
        # extract layer of optical flow from video
        opt_flows = video[..., 3]
        # sum of optical flow magnitude of individual frame
        magnitude = np.sum(opt_flows, axis=0)
        # filter slight noise by threshold
        thresh = np.mean(magnitude)
        magnitude[magnitude < thresh] = 0
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
        x = max(56, min(x, 167))
        y = max(56, min(y, 167))
        # get cropped video
        return video[:, x - 56:x + 56, y - 56:y + 56, :]
    
    def load_data(self, path):
        try:
            # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
            data = np.load(path, mmap_mode='r', allow_pickle=True)[...,3:]
            data = data.astype(np.float32)

            # sampling 64 frames uniformly from the entire video
            data = self.uniform_sampling(video=data, target_frames=64)

            # normalize rgb images and optical flows, respectively
            data = self.normalize(data)
            #print(data.shape)
            return data
        
        except Exception as e:
            # If there's an error loading the file, print an error message and return None
            print(f"Error loading file: {path}. Error message: {e}")
            return None
        
      

from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply
from tensorflow.keras.regularizers import l2

inputs = Input(shape=(64, 224, 224, 2))

#####################################################
opt = inputs
opt = Conv3D(
    16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = Conv3D(
    16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

opt = Conv3D(
    16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = Conv3D(
    16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

opt = Conv3D(
    32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = Conv3D(
    32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

opt = Conv3D(
    32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = Conv3D(
    32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    opt)
opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

#####################################################
x = MaxPooling3D(pool_size=(8, 1, 1))(opt)

#####################################################
x = Conv3D(
    64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)

x = Conv3D(
    64, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    64, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 2, 2))(x)

x = Conv3D(
    128, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = Conv3D(
    128, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
x = MaxPooling3D(pool_size=(2, 3, 3))(x)

#####################################################
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
pred = Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=pred)

model.summary()

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler


def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)
callbacks_list = [reduce_lr]

from tensorflow.keras.optimizers import Adam, SGD

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 30
num_workers = 16
batch_size = 2#16

dataset = 'RWF2000-opt'
directory_train = '../../Dataset/Numpy_Images/train'
directory_val = '../../Dataset/Numpy_Images/val'

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