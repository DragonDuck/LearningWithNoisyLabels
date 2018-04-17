# This module contains the workflow

import Data
import Model


def train_network(images, labels, model):
    """
    Train the network on the given data+labels with the given model
    :param images:
    :param labels:
    :param model:
    :return:
    """

    # TODO: DEV
    images, labels = Data.get_data(
        src="circlesquare", num_images=100000, img_dims=25)
