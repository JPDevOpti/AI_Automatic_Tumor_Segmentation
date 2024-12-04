import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session

clear_session()
def conv_block(x, filters, kernel_size, strides, padding='same'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def identity_block(x, filters):
    shortcut = x
    x = conv_block(x, filters=filters, kernel_size=(1, 1), strides=(1, 1))
    x = conv_block(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
    x = Conv2D(filters=filters * 4, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def projection_block(x, filters, strides):
    shortcut = x
    x = conv_block(x, filters=filters, kernel_size=(1, 1), strides=strides)
    x = conv_block(x, filters=filters, kernel_size=(3, 3), strides=(1, 1))
    x = Conv2D(filters=filters * 4, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters=filters * 4, kernel_size=(1, 1), strides=strides)(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def res_net_50_encoder(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = projection_block(x, filters=64, strides=(1, 1))
    x = identity_block(x, filters=64)
    x = identity_block(x, filters=64)
    x = projection_block(x, filters=128, strides=(2, 2))
    x = identity_block(x, filters=128)
    x = identity_block(x, filters=128)
    x = identity_block(x, filters=128)
    x = projection_block(x, filters=256, strides=(2, 2))
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = identity_block(x, filters=256)
    x = projection_block(x, filters=512, strides=(2, 2))
    x = identity_block(x, filters=512)
    x = identity_block(x, filters=512)
    return Model(inputs, x)

def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, filters, (3, 3), (1, 1))
    x = conv_block(x, filters, (3, 3), (1, 1))
    return x

def res_unet(input_shape=(224, 224, 3), num_classes=4, skip_layer_names=None):
    # Crear el encoder
    encoder = res_net_50_encoder(input_shape)
    encoder.summary()
    
    inputs = encoder.input
    
    # Determinar los skips
    if skip_layer_names is None:
        # Si no se especifican, usar valores predeterminados
        skip_layer_names = ["conv2d_2", "conv2d_15", "conv2d_34"]
    
    # Obtener las salidas de las capas de skip
    skips = [encoder.get_layer(name).output for name in skip_layer_names]
    
    x = encoder.output
    # Decodificador con los skips
    x = decoder_block(x, skips[2], 256)
    x = decoder_block(x, skips[1], 128)
    x = decoder_block(x, skips[0], 64)
    
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 64, (3, 3), (1, 1))
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    return Model(inputs, x)

model = res_unet(input_shape=(224, 224, 3), num_classes=4)
model.summary()


