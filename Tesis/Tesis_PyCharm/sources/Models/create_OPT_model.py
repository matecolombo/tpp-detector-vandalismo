from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.layers import Multiply  # BatchNormalization, Activation, LeakyReLU, Add,
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

# extract the rgb images
def get_rgb(input_x):
    input_rgb = input_x[..., :3]
    return input_rgb

# extract the optical flows
def get_opt(input_x):
    input_opt = input_x[..., 3:5]
    return input_opt




# Custom Loss Function
def custom_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred, from_logits=False) #True)




def create_model(inputs):
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
    return model



def create_son_model(inputs):
    opt = inputs
    opt = Conv3D(
        16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_9')(
        opt)
    opt = Conv3D(
        16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_10')(
        opt)
    opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

    opt = Conv3D(
        16, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_11')(
        opt)
    opt = Conv3D(
        16, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_12')(
        opt)
    opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

    opt = Conv3D(
        32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_13')(
        opt)
    opt = Conv3D(
        32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_14')(
        opt)
    opt = MaxPooling3D(pool_size=(1, 2, 2))(opt)

    opt = Conv3D(
        32, kernel_size=(1, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_15')(
        opt)
    opt = Conv3D(
        32, kernel_size=(3, 1, 1), strides=(1, 1, 1), kernel_initializer='he_normal', activation='relu', padding='same', name='conv3d_16')(
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
    return model