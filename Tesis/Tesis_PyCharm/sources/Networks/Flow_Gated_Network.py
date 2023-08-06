import os
import cv2
import numpy as np


#####################################################
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model  # Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import Multiply  # BatchNormalization, Activation, LeakyReLU, Add, 
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import SGD  # Adam
import tensorflow.keras.backend as backend
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger  # ModelCheckpoint,
import tensorflow.keras as keras

from tensorflow.keras.mixed_precision import global_policy, set_global_policy, Policy
import tensorflow_model_optimization as tfmot

# Definir el tipo de precisión mixta deseada (puede ser 'float16' o 'bfloat16')
policy = global_policy()
if policy.name == 'float32':
    policy = Policy('mixed_float16')
set_global_policy(policy)

#os.environ['TF_GPU_ALLOCATOR_MAX_ALLOC_PERCENT'] = '100'

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
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

# noinspection PyAttributeOutsideInit
class DataGenerator(Sequence):
    """Data Generator inherited from keras.utils.Sequence
    Args:
        directory: the path of data set, and each sub-folder will be assigned to one class
        batch_size_data: the number of data points in each batch
        shuffle: whether to shuffle the data per epoch
    Note:
        If you want to load file with other data format, please fix the method of "load_data" as you want
    """

    def __init__(self, directory, batch_size_data=1, shuffle=True, data_augmentation=True):
        # Initialize the params
        self.batch_size = batch_size_data
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.x_path, self.y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()

    def search_data(self):
        x_path = []
        y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        one_hots = utils.to_categorical(range(len(self.dirs)))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # append each file path, and keep its label
                x_path.append(file_path)
                y_dict[file_path] = one_hots[i]
        return x_path, y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.x_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.x_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
        for i, label in enumerate(self.dirs):
            print('%10s : ' % label, i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.x_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexes of each batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexes to get path of current batch
        batch_path = [self.x_path[k] for k in batch_indexes]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(d) for d in batch_path]
        batch_y = [self.y_dict[b] for b in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    @staticmethod
    def normalize(data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    @staticmethod
    def random_flip(video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video

    @staticmethod
    def uniform_sampling(video, target_frames=64):
        # get total frames of input video and calculate sampling interval
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(video[i])
                except IndexError:
                    padding.append(video[0])
            sampled_video += padding
            # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    @staticmethod
    def random_clip(video, target_frames=64):
        start_point = np.random.randint(len(video) - target_frames)
        return video[start_point:start_point + target_frames]

    @staticmethod
    def dynamic_crop(video):
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
        video_x = int(np.mean(x_points))
        video_y = int(np.mean(y_points))
        # avoid to beyond boundaries of array
        video_x = max(56, min(video_x, 167))
        video_y = max(56, min(video_y, 167))
        # get cropped video
        return video[:, video_x - 56:video_x + 56, video_y - 56:video_y + 56, :]

    # noinspection PyUnresolvedReferences
    @staticmethod
    def color_jitter(video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2, 0.2)
        v_jitter = np.random.uniform(-30, 30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 1] = 1
            v[v < 0] = 0
            v[v > 255] = 255
            hsv[..., 1] = s
            hsv[..., 2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video

    def load_data(self, path):
        try:
            # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
            data = np.load(path, mmap_mode='r', allow_pickle=True)
            data = data.astype(np.float_)
            # data = np.float32(data)
            # sampling 64 frames uniformly from the entire video
            data = self.uniform_sampling(video=data, target_frames=64)
            # whether to utilize the data augmentation
            if self.data_aug:
                data[..., :3] = self.color_jitter(data[..., :3])
                data = self.random_flip(data, prob=0.5)
            # normalize rgb images and optical flows, respectively
            data[..., :3] = self.normalize(data[..., :3])
            data[..., 3:] = self.normalize(data[..., 3:])
            return data
        except Exception as e:
            # If there's an error loading the file, print an error message and return None
            print(f"Error loading file: {path}. Error message: {e}")
            return None


#######################################################

# extract the rgb images
def get_rgb(input_x):
    input_rgb = input_x[..., :3]
    return input_rgb


# extract the optical flows
def get_opt(input_x):
    input_opt = input_x[..., 3:5]
    return input_opt


inputs = Input(shape=(64, 224, 224, 5))

rgb = Lambda(get_rgb, output_shape=None)(inputs)
opt = Lambda(get_opt, output_shape=None)(inputs)

# RGB channel
rgb = Conv3D(
    16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = Conv3D(
    16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

rgb = Conv3D(
    16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = Conv3D(
    16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

rgb = Conv3D(
    32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = Conv3D(
    32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

rgb = Conv3D(
    32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = Conv3D(
    32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same')(
    rgb)
rgb = MaxPooling3D(pool_size=(1, 2, 2))(rgb)

# Optical Flow channel
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
    32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(
    opt)
opt = Conv3D(
    32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='sigmoid', padding='same')(
    opt)
opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

# Fusion and Pooling
x = Multiply()([rgb, opt])
x = MaxPooling3D(pool_size=(8, 1, 1))(x)

# Merging Block
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

# FC Layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)

# Build the model
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=inputs, outputs=prediction)
model.summary()
'''
 Red neuronal de dos capas densas (fully connected) con una capa de salida Softmax.
 La entrada (inputs) se define previamente y se alimenta a través de una capa oculta (hidden layer) x.
 El resumen del modelo se puede obtener mediante el método summary() de la clase Model de Keras.
'''

print("Training using single GPU or CPU..")


# Model Compiling
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


# Set Callbacks
# Learning Rate Scheduler


def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = backend.get_value(model.optimizer.learning_rate)
        backend.set_value(model.optimizer.learning_rate, lr * 0.7)
    return backend.get_value(model.optimizer.learning_rate)


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

# Model Training
# set essential params
num_epochs = 30
num_workers = 16
batch_size = 2#8

initial_sparsity = 0.0
final_sparsity = 0.5
begin_step = 2000
end_step=4000
power=3  
frequency=100

dataset = 'ViolentFlow-opt'

train_generator = DataGenerator(directory='../../Dataset/Numpy_Images/train'.format(dataset), 
                                batch_size_data=batch_size,
                                data_augmentation=True)
val_generator = DataGenerator(directory='../../Dataset/Numpy_Images/val'.format(dataset),
                              batch_size_data=batch_size,
                              data_augmentation=False)

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




