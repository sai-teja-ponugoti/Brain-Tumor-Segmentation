from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add


class UNetModel:

    def __init__(self, input_size, n_filters=16, dropout=0.1, batchnorm=True, pooling_size=2, kernel_size=3):
        self.input_size = input_size
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.pooling_size = pooling_size
        self.kernel_size = kernel_size

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it
        :param input_tensor: input layer
        :param n_filters: number of filters to be used in Conv layers
        :param kernel_size: Kernel size to be used in conv layers
        :param batchnorm: boolean variable to enable or disable batchnorm
        :return: pair of conv layers as part of u-net architecture
        """
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                   padding='same')(input_tensor)
        # if batchnorm:
        #     x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
                   padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def get_unet(self):
        """Function to define the UNET Model"""
        # Contracting Path
        inputs = Input(self.input_size)

        c1 = self.conv2d_block(inputs, self.n_filters * 1, kernel_size=self.kernel_size, batchnorm=self.batchnorm)
        p1 = MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size))(c1)

        c2 = self.conv2d_block(p1, self.n_filters * 2, kernel_size=self.kernel_size, batchnorm=self.batchnorm)
        p2 = MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size))(c2)

        c3 = self.conv2d_block(p2, self.n_filters * 4, kernel_size=self.kernel_size, batchnorm=self.batchnorm)
        p3 = MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size))(c3)

        c4 = self.conv2d_block(p3, self.n_filters * 8, kernel_size=self.kernel_size, batchnorm=self.batchnorm)
        p4 = MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size))(c4)

        c5 = self.conv2d_block(p4, self.n_filters * 16, kernel_size=self.kernel_size, batchnorm=self.batchnorm)

        # Expansive Path
        up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5), c4], axis=3)
        c6 = self.conv2d_block(up6, self.n_filters * 8, kernel_size=self.kernel_size, batchnorm=self.batchnorm)

        up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6), c3], axis=3)
        c7 = self.conv2d_block(up7, self.n_filters * 4, kernel_size=self.kernel_size, batchnorm=self.batchnorm)

        up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7), c2], axis=3)
        c8 = self.conv2d_block(up8, self.n_filters * 2, kernel_size=self.kernel_size, batchnorm=self.batchnorm)

        up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8), c1], axis=3)
        c9 = self.conv2d_block(up9, self.n_filters * 1, kernel_size=self.kernel_size, batchnorm=self.batchnorm)

        c10 = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(inputs=[inputs], outputs=[c10])
