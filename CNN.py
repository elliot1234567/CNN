# Convolutional Neural Network tool
# This program will implement a CNN algorithm
# soft code dataframe and parameters for universal use

import numpy
import pandas
import tensorflow
import os

from PIL import Image

from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

from sklearn.model_selection import train_test_split

class ConvolutionalNeuralNetwork():

    model = Sequential()
    classes = []
    classDictionary = {}
    dataframe = None
    X = None
    y = None
    score = None
    path = None

    def __init__(self, directory):
        """
            Constructs a new CNN object.

            Parameters
            ----------
            directory : string[]
                path to folder containing folders with data in them
        """
        count = 0
        self.path = directory
        for d in os.listdir(directory):
            self.classes += d
            self.classDictionary[d] = count
            count+=1

        count = 0
        self.dataframe = pandas.DataFrame(columns=["class", "image"])
        for d in os.listdir(directory):
            folder = d
            d = self.path + r"\\" + d
            for image in os.listdir(d):
                image = d + r"\\" + image
                img = Image.open(image).convert('L')
                data = numpy.asarray(img)
                self.dataframe.at[count, "image"] = data
                self.dataframe.at[count, "class"] = self.classDictionary[folder]
                count+=1
            
    def separateVariables(self):
        """
            Separates features from dataframe into
            independent and dependent variables.
        """
        self.X = self.dataframe.drop("class", axis=1)
        self.y = self.dataframe[["class"]]

    def processModel(self, split, random):
        """
            Trains and scores model from train and test data.

            Parameters:
            -----------
            split : int
                the decimal value between 0 and 1 for the amount
                of test data to separate from the training data
            random : int
                random state value
        """
        x_train,x_test,y_train,y_test = train_test_split(self.X, self.y, test_size=split, random_state=random)

        x_train = tensorflow.convert_to_tensor(x_train, dtype=tensorflow.int64)
        x_test = tensorflow.convert_to_tensor(x_test, dtype=tensorflow.int64)
        y_train = tensorflow.convert_to_tensor(y_train, dtype=tensorflow.int64)
        y_test = tensorflow.convert_to_tensor(y_test, dtype=tensorflow.int64)

        self.model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        self.model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
        self.score = test_acc

    def addConvolutionLayer(
                            self,
                            filters,
                            kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            data_format=None,
                            dilation_rate=(1, 1),
                            groups=1,
                            activation=None,
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None
        ):
        self.model.add(Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        groups=groups,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint
        ))
    def addBatchNormalizationLayer(
                                    self,
                                    axis=-1,
                                    momentum=0.99,
                                    epsilon=0.001,
                                    center=True,
                                    scale=True,
                                    beta_initializer='zeros',
                                    gamma_initializer='ones',
                                    moving_mean_initializer='zeros',
                                    moving_variance_initializer='ones',
                                    beta_regularizer=None,
                                    gamma_regularizer=None,
                                    beta_constraint=None,
                                    gamma_constraint=None,
                                    renorm=False,
                                    renorm_clipping=None,
                                    renorm_momentum=0.99,
                                    fused=None,
                                    trainable=True,
                                    virtual_batch_size=None,
                                    adjustment=None,
                                    name=None
        ):
        self.model.add(BatchNormalization(
                                    axis=axis,
                                    momentum=momentum,
                                    epsilon=epsilon,
                                    center=center,
                                    scale=scale,
                                    beta_initializer=beta_initializer,
                                    gamma_initializer=gamma_initializer,
                                    moving_mean_initializer=moving_mean_initializer,
                                    moving_variance_initializer=moving_variance_initializer,
                                    beta_regularizer=beta_regularizer,
                                    gamma_regularizer=gamma_regularizer,
                                    beta_constraint=beta_constraint,
                                    gamma_constraint=gamma_constraint,
                                    renorm=renorm,
                                    renorm_clipping=renorm_clipping,
                                    renorm_momentum=renorm_momentum,
                                    fused=fused,
                                    trainable = trainable,
                                    virtual_batch_size=virtual_batch_size,
                                    adjustment=adjustment,
                                    name=name
        ))
    def addMaxPoolingLayer(
                            self,
                            pool_size=(2, 2),
                            strides=None,
                            padding='valid',
                            data_format=None
        ):
        self.model.add(MaxPooling2D(
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                            data_format=data_format
        ))
    def addDropOutLayer(
                        self,
                        rate,
                        noise_shape=None,
                        seed=None
        ):
        self.model.add(Dropout(
                        rate=rate,
                        noise_shape=noise_shape,
                        seed=seed
        ))
    def addFlattenLayer(
                        self,
                        data_format=None
        ):
        self.model.add(Flatten(
                        data_format=data_format
        ))
    def addDenseLayer(
                        self,
                        units,
                        activation=None,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                        ):
        self.model.add(Dense(
                        units=units,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint
        ))

    def getScore(self):
        return self.score