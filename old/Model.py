# This module defines a keras module

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten


def get_model(
        img_shape, nclasses, model_type="convolutional", **kwargs):
    """
    Build a keras model
    :param img_shape:
    :param nclasses:
    :param model_type:
    :param kwargs: Additional parameters to be passed to
           keras.models.Model.compile(...). Default values are
           loss="*_crossentropy" ("binary" for nclasses=2 and
           "categorical" for nclasses>2) and optimizer="Adam"
    :return:
    """

    if model_type.lower().startswith("c"):  # convolutional
        model = build_convnet(img_shape=img_shape, nclasses=nclasses)
    elif model_type.lower().startswith("f"):  # fully-connected
        model = build_fcnet(img_shape=img_shape, nclasses=nclasses)
    else:
        raise ValueError("'model_type' unknown")

    if "loss" not in kwargs.keys():
        loss = "binary_crossentropy" if nclasses == 2 \
            else "categorical_crossentropy"
    else:
        loss = kwargs["loss"]

    if "optimizer" not in kwargs.keys():
        optimizer = "Adam"
    else:
        optimizer = kwargs["optimizer"]

    model.compile(loss=loss, optimizer=optimizer)

    return model


def build_convnet(img_shape, nclasses):
    """
    Build convolutional network
    :param img_shape:
    :param nclasses:
    :return:
    """
    inputs = Input(shape=img_shape)
    conv1 = Conv2D(
        filters=32, kernel_size=(3, 3),
        activation="relu", padding="valid",
        strides=(1, 1))(inputs)
    conv2 = Conv2D(
        filters=32, kernel_size=(3, 3),
        activation="relu", padding="valid",
        strides=(1, 1))(conv1)
    maxpool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv2)

    conv3 = Conv2D(
        filters=64, kernel_size=(3, 3),
        activation="relu", padding="valid",
        strides=(1, 1))(maxpool1)
    conv4 = Conv2D(
        filters=64, kernel_size=(3, 3),
        activation="relu", padding="valid",
        strides=(1, 1))(conv3)
    maxpool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(conv4)

    flattened = Flatten()(maxpool2)

    outputs = Dense(units=nclasses, activation="sigmoid")(flattened)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def build_fcnet(img_shape, nclasses):
    inputs = Input(shape=img_shape)
    fc1 = Dense(units=128)(inputs)
    outputs = Dense(units=nclasses)(fc1)

    model = Model(inputs=inputs, outputs=outputs)
    return model
