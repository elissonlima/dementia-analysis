from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D
import numpy as np


class DementiaAnalysisModel(object):

    def __init__(self, input_shape=(248, 320, 1)):
        """
            Demetia Analysis Model with CNN model init

            Arguments:
            - input_shape : (tuple) shape of input image
        """
        np.random.seed(1)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=3, strides=2,
                     activation='sigmoid', input_shape=input_shape))
        self.model.add(AveragePooling2D(pool_size=(2,2), strides=2))
        self.model.add(Conv2D(16, kernel_size=3, strides=2,
                     activation='sigmoid'))
        self.model.add(AveragePooling2D(pool_size=(2,2), strides=2))
        self.model.add(Conv2D(8, kernel_size=3, strides=2,
                     activation='sigmoid'))
        self.model.add(AveragePooling2D(pool_size=(2,2), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(150, activation='sigmoid'))
        self.model.add(Dense(10, activation='sigmoid'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='RMSprop', loss='binary_crossentropy', 
            metrics=['accuracy'])

    def train(self, dataset, labels, epochs):
        self.model.fit(dataset, labels,
            epochs=epochs)