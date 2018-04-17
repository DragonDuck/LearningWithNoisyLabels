# This module is a data API

import numpy as np
import h5py
import os


def generate_circlesquare(num_images=100000, img_dims=25):
    """
    This function generates an HDF5 file with circles and squares. If a data
    file with the given specifications already exists, it is not overwritten.

    The images themselves are given some noise in the form of:
        - varying mean intensities
        - varying sizes
        - Salt & Pepper noise

    The images generated are 8bit, i.e. the intensities have values in the
    range [0, 255].

    The hdf5 file format is
    HDF5:
        - Dataset 'images' (num_images, 1, img_dims, img_dims)
        - Dataset 'labels' (num_images, 1)

    :param num_images:
    :param img_dims:
    :return:
    """
    if img_dims % 2 == 0:
        img_dims += 1
        print(
            "'img_dims' must be odd and has been "
            "set to img_dims = {}".format(img_dims))
    if img_dims < 11:
        raise ValueError(
            "'img_dims' must be at least 11 for a differentiation between "
            "circles and squares to make sense.")

    h5fn = "CircleSquareImages_numimgs-{}_size-{}px.h5".format(
        num_images, img_dims)
    if os.path.isfile(h5fn):
        print("File '{}' exists!".format(h5fn))
        return None

    max_rad = ((img_dims - 1) // 2) - 1
    center = (img_dims - 1) // 2
    ind_y, ind_x = np.indices(dimensions=(img_dims, img_dims)) - center

    # Circles
    rand_rads = np.random.randint(5, max_rad+1, num_images)
    rand_intensities = np.random.randint(1, 256, num_images)
    noise_field = np.random.randint(-32, 33, (img_dims, img_dims, num_images))
    rad_field = np.repeat(np.expand_dims(np.sqrt(
        ind_y**2 + ind_x**2), 2), num_images, 2) <= rand_rads
    img_circles = (rad_field.astype(dtype=np.int16) * rand_intensities)
    img_circles += noise_field
    img_circles[img_circles < 0] = 0
    img_circles[img_circles > 255] = 255

    # Squares
    rand_rads = np.random.randint(5, max_rad+1, num_images)
    rand_intensities = np.random.randint(1, 256, num_images)
    noise_field = np.random.randint(-32, 33, (img_dims, img_dims, num_images))
    ind_x_square = np.repeat(np.expand_dims(ind_x, 2), num_images, 2)
    ind_y_square = np.repeat(np.expand_dims(ind_y, 2), num_images, 2)
    square_field = (np.abs(ind_x_square) <= rand_rads) * \
                   (np.abs(ind_y_square) <= rand_rads)
    img_squares = (square_field.astype(dtype=np.int16) * rand_intensities)
    img_squares += noise_field
    img_squares[img_squares < 0] = 0
    img_squares[img_squares > 255] = 255

    imgs = np.concatenate((img_circles, img_squares), 2)
    imgs = np.transpose(imgs, (2, 0, 1))
    imgs = np.expand_dims(imgs, 1)
    imgs = imgs.astype(np.uint8)
    labels = np.expand_dims(np.repeat(("Circle", "Square"), (num_images, num_images)), 1)

    with h5py.File(h5fn, "w-") as h5handle:
        h5handle.create_dataset(
            name="images", data=imgs,
            chunks=(1, 1, img_dims, img_dims),
            compression=3)
        h5handle.create_dataset(
            name="labels", data=labels.astype(np.bytes_),
            chunks=(1, 1), compression=3)


def get_data(src="circlesquare", **kwargs):
    """
    Loads data from hdf5.

    If loading 'circlesquare', then 'num_images' and 'img_dims' must be passed
    as keyword parameters.
    :param src:
    :return:
    """
    if src.lower().startswith("c"):  # circlesquare
        num_images = kwargs["num_images"]
        img_dims = kwargs["img_dims"]
        h5fn = "CircleSquareImages_numimgs-{}_size-{}px.h5".format(
            num_images, img_dims)
    elif src.lower().startswith("m"): # mnist
        h5fn = "MNIST.h5"
    else:
        raise ValueError("'src' not supported")

    with h5py.File(h5fn, "r") as h5handle:
        images = h5handle["images"][()]
        labels = h5handle["labels"][()].astype(np.str)

    return images, labels
