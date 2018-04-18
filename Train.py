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

    sel_indices = np.in1d(labels, ("1", "2", "9"))
    images = images[sel_indices, ...]
    labels = labels[sel_indices]

    # Loop through scrambling percentages
    # Make sure the first entry stays 0 as there is no other record of the
    # DNNs true performance
    # percentages = np.array([
    #     0, 0.25, 0.5, 0.75, 0.8, 0.825,
    #     0.85, 0.875, 0.9, 0.95, 1.0])
    percentages = np.array([
         0, 0.2, 0.4, 0.45, 0.5, 0.6])
    true_scores = []
    scrambled_scores = []
    predicted_labels = []
    for p in percentages:
        print("Scrambling = {}".format(p))

        true_score, scrambled_score, model = train_and_score_network(
            dataset=dataset, images=images, labels=labels,
            percentage=p, modeltype=modeltype)
        true_scores.append(true_score)
        scrambled_scores.append(scrambled_score)

        pred = model.predict(x=images)
        pred = np.argmax(pred, axis=1)
        label_dict = get_label_dict(dataset=dataset)
        reverse_label_dict = {val: key for key, val in label_dict.items()}
        pred_str = np.array([
            reverse_label_dict[pred[ii]] for ii
            in range(len(pred))])
        predicted_labels.append(pred_str)

        print("Scrambling = {}: True Score = {}; Scrambled Score = {}".format(
            p, true_score, scrambled_score))

    results = pd.DataFrame(
        data={
            "TrueF1": true_scores,
            "ScrambledF1": scrambled_scores},
        index=percentages)
    outfn = "ResultsScrambling_ThreeClasses_{}_{}.csv".format(
        dataset, modeltype)
    results.to_csv(outfn)

    pred_labels = pd.DataFrame(
        data=np.stack(predicted_labels, axis=1),
        columns=["Iter{}".format(ii) for ii in range(len(predicted_labels))])
    pred_labels["TrueLabels"] = labels

    ploutfn = "PredictionsScrambling_ThreeClasses_{}_{}.csv".format(
        dataset, modeltype)
    pred_labels.to_csv(ploutfn)


def workflow_scrambling_multiple_iterations(dataset="mnist", modeltype="conv"):
    """
    This workflow tests different scrambling degrees
    :param dataset:
    :param modeltype:
    :return:
    """
    images, labels = Data.get_data(src=dataset)
    label_dict = get_label_dict(dataset=dataset)
    reverse_label_dict = {val: key for key, val in label_dict.items()}

    # Loop through scrambling percentages
    # Make sure the first entry stays 0 as there is no other record of the
    # DNNs true performance
    percentages = np.array([
        0, 0.25, 0.5, 0.75, 0.8, 0.825,
        0.85, 0.875, 0.9, 0.95, 1.0])
    true_scores = []
    scrambled_scores = []
    predicted_labels = []
    for p in percentages:
        print("Scrambling = {}".format(p))

        iter_true_scores = []
        iter_scrambled_scores = []
        iter_pred = []
        for ii in range(10):
            true_score, scrambled_score, model = train_and_score_network(
                dataset=dataset, images=images, labels=labels,
                percentage=p, modeltype=modeltype)
            iter_true_scores.append(true_score)
            iter_scrambled_scores.append(scrambled_score)

            pred = model.predict(x=images)
            pred = np.argmax(pred, axis=1)
            pred_str = np.array([
                reverse_label_dict[pred[ii]] for ii
                in range(len(pred))])
            iter_pred.append(pred_str)

        true_scores.append(np.stack(iter_true_scores))
        scrambled_scores.append(np.stack(iter_scrambled_scores))
        predicted_labels.append(np.stack(iter_pred))

        print("Scrambling = {}: True Score = {}; Scrambled Score = {}".format(
            p, np.mean(iter_true_scores), np.mean(iter_scrambled_scores)))

    np.save("TrueScores.npy", np.stack(true_scores))
    np.save("ScrambledScores.npy", np.stack(scrambled_scores))
    np.save("PredictedLabels.npy", np.stack(predicted_labels))


def workflow_iterative_training(
        scrambling=0.9, dataset="mnist", modeltype="conv"):
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
    while len(scores) <= 10:
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
    iteration n, labels are only changed for the 10% most certain new labels
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
    while len(scores) < 10:
        # Quit if early convergence
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

        # Update labels for the most certain entries
        pred_full = model.predict(x=images)
        pred_full_sm = np.exp(pred_full) / np.sum(
            np.exp(pred_full), axis=1)[:, None]

        new_certainty = np.max(pred_full_sm, axis=1)
        new_label = np.argmax(pred_full_sm, axis=1)
        old_label = np.argmax(cur_labels, axis=1)
        certainty_thresh = np.percentile(
            a=new_certainty[new_label != old_label], q=90)
        # Set cur_label with new_thresh > certainty_thresh to the
        # new label
        old_label[new_certainty >= certainty_thresh] = new_label[
            new_certainty >= certainty_thresh]
        cur_labels = keras.utils.to_categorical(old_label)

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
