# This module contains the workflow

import Data
import Model
import numpy as np
import keras.utils
import sklearn.model_selection
import sklearn.metrics
import pandas as pd


def workflow_scrambling(dataset="mnist", modeltype="conv"):
    """
    This workflow tests different scrambling degrees
    :param dataset:
    :param modeltype:
    :return:
    """
    images, labels = Data.get_data(src=dataset)

    # Loop through scrambling percentages
    percentages = np.array([
        0, 0.25, 0.5, 0.75, 0.8, 0.825,
        0.85, 0.875, 0.9, 0.95, 1.0])
    true_scores = []
    scrambled_scores = []
    for p in percentages:
        print("Scrambling = {}".format(p))
        true_score, scrambled_score, model = train_and_score_network(
            dataset=dataset, images=images, labels=labels,
            percentage=p, modeltype=modeltype)
        true_scores.append(true_score)
        scrambled_scores.append(scrambled_score)
        print("Scrambling = {}: True Score = {}; Scrambled Score = {}".format(
            p, true_score, scrambled_score))

    results = pd.DataFrame(
        data={
            "TrueF1": true_scores,
            "ScrambledF1": scrambled_scores},
        index=percentages)

    outfn = "ResultsScrambling_{}_{}.csv".format(dataset, modeltype)
    results.to_csv(outfn)


def workflow_iterative_training(
        scrambling=0.8, dataset="mnist", modeltype="conv"):
    """
    This workflow tests iterative training for a fixed scaling.

    In this workflow, the labels are scrambled once at the beginning and then
    a reconstruction is attempted.
    :param scrambling
    :param dataset:
    :param modeltype:
    :return:
    """
    images, labels = Data.get_data(src=dataset)

    # Scramble labels
    cur_labels = scramble_labels(
        labels=labels, percentage=scrambling)

    # Turn labels into one-hot vectors
    label_dict = get_label_dict(dataset=dataset)
    labels = np.stack([
        label_dict[labels[ii]]
        for ii in range(len(labels))])
    labels = keras.utils.to_categorical(labels)
    cur_labels = np.stack([
        label_dict[cur_labels[ii]]
        for ii in range(len(cur_labels))])
    cur_labels = keras.utils.to_categorical(cur_labels)

    # Get initial f1-score
    scores = [sklearn.metrics.f1_score(
        y_true=np.argmax(labels, axis=1),
        y_pred=np.argmax(cur_labels, axis=1),
        average="weighted")]

    # Loop until reconstructed or until abort criterion
    while True:
        # Abort criterion
        if len(scores) > 10:
            print("WARNING: No convergence after 10 iterations")
            break

        # Quit if f1-score is close to 1
        if scores[-1] > 0.95:
            break

        print("Iteration {}".format(len(scores)))

        # Split into train and test
        split = sklearn.model_selection.train_test_split(
            images, labels, cur_labels, test_size=0.5)
        images_train, images_test = split[0:2]
        labels_train, labels_test = split[2:4]
        cur_labels_train, cur_labels_test = split[4:6]

        # Train network on current labels
        model = Model.get_model(
            img_shape=images.shape[1:4],
            nclasses=labels.shape[1],
            model_type=modeltype)
        model.fit(
            x=images_train, y=cur_labels_train,
            batch_size=32, epochs=5, verbose=2)

        # Calculate f1-score
        pred = model.predict(x=images_test)
        pred = np.argmax(pred, axis=1)
        scores.append(sklearn.metrics.f1_score(
            y_true=pred,
            y_pred=np.argmax(labels_test, axis=1),
            average="weighted"))

        # Update labels
        pred_full = model.predict(x=images)
        pred_full = np.argmax(pred_full, axis=1)
        cur_labels = keras.utils.to_categorical(pred_full)

        print("Iteration {}: Score = {}".format(
            len(scores) - 1, scores[-1]))

    results = pd.DataFrame(
        data={"TrueF1": scores},
        index=np.arange(0, len(scores)))

    outfn = "ResultsIterative_{}_{}_{}.csv".format(
        dataset, modeltype, scrambling)
    results.to_csv(outfn)


def workflow_strict_iterative_training(
        scrambling=0.8, dataset="mnist", modeltype="conv"):
    """
    This workflow tests iterative training for a fixed scaling.

    In this workflow, the labels are scrambled once at the beginning and then
    a reconstruction is attempted.

    Instead of naively using the output of iteration n-1 as labels for
    iteration n, only those labels with a new certainty of >= 95% are changed.
    :param scrambling
    :param dataset:
    :param modeltype:
    :return:
    """
    images, labels = Data.get_data(src=dataset)

    # Scramble labels
    cur_labels = scramble_labels(
        labels=labels, percentage=scrambling)

    # Turn labels into one-hot vectors
    label_dict = get_label_dict(dataset=dataset)
    labels = np.stack([
        label_dict[labels[ii]]
        for ii in range(len(labels))])
    labels = keras.utils.to_categorical(labels)
    cur_labels = np.stack([
        label_dict[cur_labels[ii]]
        for ii in range(len(cur_labels))])
    cur_labels = keras.utils.to_categorical(cur_labels)

    # Get initial f1-score
    scores = [sklearn.metrics.f1_score(
        y_true=np.argmax(labels, axis=1),
        y_pred=np.argmax(cur_labels, axis=1),
        average="weighted")]

    # Loop until reconstructed or until abort criterion
    while True:
        # Abort criterion
        if len(scores) > 10:
            print("WARNING: No convergence after 10 iterations")
            break

        # Quit if f1-score is close to 1
        if scores[-1] > 0.95:
            break

        print("Iteration {}".format(len(scores)))

        # Split into train and test
        split = sklearn.model_selection.train_test_split(
            images, labels, cur_labels, test_size=0.5)
        images_train, images_test = split[0:2]
        labels_train, labels_test = split[2:4]
        cur_labels_train, cur_labels_test = split[4:6]

        # Train network on current labels
        model = Model.get_model(
            img_shape=images.shape[1:4],
            nclasses=labels.shape[1],
            model_type=modeltype)
        model.fit(
            x=images_train, y=cur_labels_train,
            batch_size=32, epochs=5, verbose=2)

        # Calculate f1-score
        pred = model.predict(x=images_test)
        pred = np.argmax(pred, axis=1)
        scores.append(sklearn.metrics.f1_score(
            y_true=pred,
            y_pred=np.argmax(labels_test, axis=1),
            average="weighted"))

        # Update labels
        pred_full = model.predict(x=images)
        pred_full = np.argmax(pred_full, axis=1)
        cur_labels = keras.utils.to_categorical(pred_full)

        print("Iteration {}: Score = {}".format(
            len(scores) - 1, scores[-1]))

    results = pd.DataFrame(
        data={"TrueF1": scores},
        index=np.arange(0, len(scores)))

    outfn = "ResultsIterative_{}_{}_{}.csv".format(
        dataset, modeltype, scrambling)
    results.to_csv(outfn)


def train_and_score_network(dataset, images, labels, percentage, modeltype="conv"):
    """
    Trains the network on the given data and a scrambled permutation of the
    labels. It then calculates the f1 score on the true labels to determine
    how well the network ignores bad labels

    Returns the trained model and the validation statistics.
    :param dataset:
    :param images: Numpy array (num_images, x, y, channels)
    :param labels: Numpy array (num_images, num_classes)
    :param percentage: Float. Truncated to [0, 1]
    :param modeltype:
    :return:
    """
    # Scramble the labels
    scrambled_labels = scramble_labels(
        labels=labels, percentage=percentage)

    # Make labels one-hot encoded
    label_dict = get_label_dict(dataset=dataset)
    labels = np.stack([
        label_dict[labels[ii]]
        for ii in range(len(labels))])
    labels = keras.utils.to_categorical(labels)
    scrambled_labels = np.stack([
        label_dict[scrambled_labels[ii]]
        for ii in range(len(scrambled_labels))])
    scrambled_labels = keras.utils.to_categorical(scrambled_labels)

    # Set up model
    model = Model.get_model(
        img_shape=images.shape[1:4],
        nclasses=labels.shape[1],
        model_type=modeltype)

    # Split into training and testing set
    split = sklearn.model_selection.train_test_split(
        images, labels, scrambled_labels, test_size=0.50)
    x_train, x_test, y_train, y_test, scr_train, scr_test = split

    # Fit model
    model.fit(x=x_train, y=scr_train, batch_size=32, epochs=5, verbose=2)

    # Predict on test set and calculate AUC
    pred = model.predict(x=x_test)

    f1score_real = sklearn.metrics.f1_score(
        y_true=np.argmax(y_test, axis=1),
        y_pred=np.argmax(pred, axis=1),
        average="weighted")

    f1score_scrambled = sklearn.metrics.f1_score(
        y_true=np.argmax(scr_test, axis=1),
        y_pred=np.argmax(pred, axis=1),
        average="weighted")

    return f1score_real, f1score_scrambled, model


def scramble_labels(labels, percentage):
    """
    Takes a label vector and randomly scrambles a fraction of the labels.

    The scrambling ensures that none of the altered labels will have their
    original value, i.e. the scrambling percentage is guaranteed.

    'labels' must be a categorical label

    :param labels: Numpy array (num_labels,)
    :param percentage: Integer
    :return:
    """
    if percentage > 1:
        print("'scramble_freq' truncated to 1.0")
        percentage = 1.0

    if percentage == 0:
        return labels

    labels = labels.flatten()

    labels_to_scramble = np.random.choice(
        a=range(len(labels)),
        size=int(len(labels) * percentage),
        replace=False)

    # Define possible values for each entry
    possible_values = np.unique(labels)
    possible_values = np.repeat(
        a=np.expand_dims(possible_values, 0),
        repeats=labels_to_scramble.shape[0],
        axis=0)
    forbidden_values = np.repeat(
        a=np.expand_dims(labels[labels_to_scramble], 1),
        repeats=len(np.unique(labels)), axis=1)

    sel = possible_values != forbidden_values
    new_possible_values = np.reshape(
        a=possible_values[sel],
        newshape=(possible_values.shape[0], possible_values.shape[1]-1))

    new_values = []
    for vals in new_possible_values:
        new_values.append(np.random.choice(vals))
    new_values = np.array(new_values)
    labels[labels_to_scramble] = new_values

    return labels


def get_label_dict(dataset):
    """
    Get a label dictionary. This is done by more than one function, so a
    consistent function call is better.
    :param dataset:
    :return:
    """
    ds_labels = Data.get_labels(src=dataset)
    return {
        np.sort(np.unique(ds_labels))[ii]: ii
        for ii in range(len(np.unique(ds_labels)))}
