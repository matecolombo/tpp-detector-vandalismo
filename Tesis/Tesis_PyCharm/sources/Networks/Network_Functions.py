import os
import cv2
import numpy as np

from tensorflow.keras import utils
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as backend
import tensorflow.keras as keras


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


# extract the rgb images
def get_rgb(input_x):
    input_rgb = input_x[..., :3]
    return input_rgb


# extract the optical flows
def get_opt(input_x):
    input_opt = input_x[..., 3:5]
    return input_opt


# Set Callbacks
# Learning Rate Scheduler
def scheduler(model, epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = backend.get_value(model.optimizer.learning_rate)
        backend.set_value(model.optimizer.learning_rate, lr * 0.7)
    return backend.get_value(model.optimizer.learning_rate)


# Saving the best model and training logs

class MyCbk(keras.callbacks.Callback):
    def __init__(self, model_to_save):
        super().__init__()
        self.model_to_save = model_to_save

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('Logs/model_at_epoch_%d.h5' % (epoch + 1))


class DataGenerator_adapted(Sequence):
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
        self.indexes = None
        self.n_files = None
        self.dirs = None
        self.batch_size = batch_size_data
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.x_path, self.y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()

    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        # one_hots = utils.to_categorical(range(len(self.dirs)))
        one_hots = utils.to_categorical(range(len(self.directory)))
        i = 0
        #        for i, folder in enumerate(self.dirs):
        folder_path = os.path.join(self.directory)  # , folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # append each file path, and keep its label
            X_path.append(file_path)
            Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.x_path)
        self.indexes = np.arange(len(self.x_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files.".format(self.n_files))
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
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r')
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




class DataGenerator_tflite(Sequence):
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
        self.indexes = None
        self.n_files = None
        self.dirs = None
        self.batch_size = batch_size_data
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.x_path, self.y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()

    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        # one_hots = utils.to_categorical(range(len(self.dirs)))
        one_hots = utils.to_categorical(range(len(self.directory)))
        i = 0
        #        for i, folder in enumerate(self.dirs):
        folder_path = os.path.join(self.directory)  # , folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            # append each file path, and keep its label
            X_path.append(file_path)
            Y_dict[file_path] = one_hots[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.x_path)
        self.indexes = np.arange(len(self.x_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files.".format(self.n_files))
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
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r')
        #data = data.astype(np.float_)
        data = data.astype(np.float32)
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
