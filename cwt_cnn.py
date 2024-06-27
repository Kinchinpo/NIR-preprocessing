import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from scipy import signal
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from keras.src import backend
from keras.src import layers
from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.models import Functional
from keras.src.ops import operation_utils
from keras.src.utils import file_utils
from keras.optimizers import Adam
from keras import metrics

file_path_nir = r'dataverse_files/spectra-1.csv'
df = pd.read_csv(file_path_nir)

wavelength = df.columns[1:].to_numpy()
wavelength = np.array([float(item[:-3]) if item.endswith('_nm') else item for item in wavelength])
absorbance = df.values[:][0:]
absorbance = np.delete(absorbance, 0, axis=1)
absorbance = np.vectorize(lambda x: float(x.replace(',', '.')))(absorbance)

file_path_manure = r'dataverse_files/chemical_analysis.xlsx'
manure_data = pd.read_excel(file_path_manure)
manure_data = manure_data.values
manure_data = manure_data[:, 3:]
# print(manure_data)
# print(manure_data.shape)
manure_type = manure_data[:, :1]
chemical_decom = manure_data[:, 5:6]

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 all_filenames, 
                 labels, 
                 batch_size, 
                 index2class,
                 input_dim,
                 n_channels,
                 n_classes=2, 
                 normalize=True,
                 zoom_range=[0.8, 1],
                 rotation=15,
                 brightness_range=[0.8, 1],
                 shuffle=True,
                 shift_range=0.1,  # Specify the shift percentage here
                 shift_directions=['left', 'right', 'up', 'down']):  # Specify the shift directions
        # Initialize class variables
        # Include shift_range and shift_directions in the class parameters
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.index2class = index2class
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.normalize = normalize
        self.zoom_range = zoom_range
        self.rotation = rotation
        self.brightness_range = brightness_range
        self.shuffle = shuffle
        self.shift_range = shift_range
        self.shift_directions = shift_directions
        self.on_epoch_end()
    
    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # List all_filenames trong một batch
        all_filenames_temp = [self.all_filenames[k] for k in indexes]

        # Khởi tạo data
        X, y = self.__data_generation(all_filenames_temp)

        return X, y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    # Previous methods remain the same
    def shift_image(self, image, shift_percent, direction):
        """
        Sets the corresponding shifted part of the input image to zero.

        Args:
            image (np.ndarray): Input image with shape (height, width, 3).
            shift_percent (float): Percentage of the image size to shift (e.g., 0.1 for 10%).
            direction (str): Direction of the shift ('left', 'right', 'up', or 'down').

        Returns:
            np.ndarray: Image with the shifted part set to zero.
        """
        if direction not in ('left', 'right', 'up', 'down'):
            raise ValueError("Invalid direction. Choose from 'left', 'right', 'up', or 'down'.")

        # Calculate the shift amount based on the image size
        shift_amount = int(shift_percent * image.shape[0])

        # Initialize the shifted image
        shifted_image = np.copy(image)

        if direction == 'left':
            shifted_image[:, -shift_amount:, :] = 0
        elif direction == 'right':
            shifted_image[:, :shift_amount, :] = 0
        elif direction == 'up':
            shifted_image[-shift_amount:, :, :] = 0
        elif direction == 'down':
            shifted_image[:shift_amount, :, :] = 0

        return shifted_image

    
    def __data_generation(self, all_filenames_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)
        for i, fn in enumerate(all_filenames_temp):
            img = cv2.imread(fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_dim)
            img_reshape = img.reshape(-1, 3)
            
            # Perform data augmentation, including image shifting
            direction = np.random.choice(self.shift_directions)
            shifted_image = self.shift_image(img, self.shift_range, direction)
            X[i,] = shifted_image
            y[i] = self.labels[i]
        
        return X, y

image_folder = r'absorbance_images/'
image_filenames = glob.glob(os.path.join(image_folder, '*.png'))  
sorted_indices = np.argsort([int(name.split('_')[-1].split('.')[0]) for name in image_filenames])
image_filenames = [image_filenames[i] for i in sorted_indices]

all_filenames = image_filenames  # X data
labels = chemical_decom  # y data
batch_size = 32
input_dim = (256, 256)
n_channels = 3
X_train, X_test, y_train, y_test = train_test_split(all_filenames, labels, test_size=.2)
X_train_new = []
X_test_new = []
for i, fn in enumerate(X_train):
    X_train_new.append(cv2.imread(fn))
    
for i, fn in enumerate(X_test):
    X_test_new.append(cv2.imread(fn))

X_train_new = np.array(X_train_new)
X_test_new = np.array(X_test_new)
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
# y_train = y_train.flatten()
# y_test = y_test.flatten()
print(X_train_new.shape)
# print(y_test.shape)

def conv2d_bn(
    x, filters, num_row, num_col, padding="same", strides=(1, 1), name=None
):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == "channels_first":
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
    )(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation("relu", name=name)(x)
    return x

def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )
    
def getModel(dropout=.25, learning_rate=.01):
    inputs = tf.keras.Input(shape=(256, 256, 3))
    
    x = inputs
    x = preprocess_input(x)
    x = conv2d_bn(x, 32, 3, 3, padding="same", strides=(1, 1), name="Layer_1")
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = conv2d_bn(x, 64, 3, 3, padding="same", strides=(1, 1), name='Layer_2')
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = conv2d_bn(x, 128, 3, 3, padding="same", strides=(1, 1), name='Layer_3')
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate),
                  metrics=['mae', 'r2_score'])

    return model

model = getModel()
model.summary()

history = model.fit(X_train_new, y_train,
                    validation_data=(X_test_new, y_test),
                    epochs=10)