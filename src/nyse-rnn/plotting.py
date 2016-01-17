import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_features():
    with open('../../results/RNN_features', 'rb') as f:
        feature_selection = pickle.load(f)

    # n = feature_selection["count"]
    n = 30
    feature_mean_errors = []
    for featureNo in range(n):
        mean_error = np.average(np.ravel([y for (x,y) in feature_selection["results"] if featureNo in x]))
        print("Feature {0} -> Mean error = {1}".format(featureNo, mean_error))
        feature_mean_errors.append(mean_error)

    print("Selected features: {0}".format(feature_selection["features"]))
    print("Feature mean errors: {0}".format(feature_mean_errors))

    features_figure = plt.figure()
    ax = plt.subplot(111)
    feature_nos = range(len(feature_mean_errors))
    ax.bar(feature_nos, feature_mean_errors, width=0.5, align='center')
    plt.xticks(feature_nos, feature_nos)
    features_figure.show()
    features_figure.waitforbuttonpress()


def rrn_iter_error_plot():
    with open('../../results/RNN_errors', 'rb') as f:
        errors = pickle.load(f)

    print("Train ERRORS: {0}".format(errors["train"]))
    print("Test ERRORS: {0}".format(errors["test"]))

    plt.figure()
    plt.title("RNN error X iteration")
    plt.plot(errors["train"])
    plt.plot(errors["test"])
    plt.xlabel("iteration * 10")
    plt.ylabel("error rate")
    plt.show()


def mlp_iter_error_plot():
    with open('../../results/MLP_errors', 'rb') as f:
        errors = pickle.load(f)

    print("Train ERRORS: {0}".format(errors["train"]))
    print("Test ERRORS: {0}".format(errors["test"]))

    plt.figure()
    plt.title("RNN error X iteration")
    plt.plot(errors["train"])
    plt.plot(errors["test"])
    plt.xlabel("iteration * 10")
    plt.ylabel("error rate")
    plt.show()


