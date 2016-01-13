import rnn
import perceptron
from nyse import *
import nn
import classification_performance as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt


def fs():
    input_length = 100
    hidden_cnt = 50
    data = get_test_data(input_length)

    rnn_nn = nn.NeuralNetwork(nn=rnn.RNN(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]),
                              validation_split=0.2,
                              batch_size=256,
                              nb_epoch=10,
                              show_accuracy=True)

    features, results = rnn_nn.feature_selection(data)
    print("Selected features: {0}".format(features))
    print(results)

    feature_selection = {"features": features, "results": results, "count": data.x.shape[2]}
    output = open('RNN_features', 'wb')
    pickle.dump(feature_selection, output)
    output.close()


def plot_features():

    with open('RNN_features', 'rb') as f:
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



def main():
    input_length = 100
    hidden_cnt = 50
    cross_validation_passes = 10

    data = get_test_data(input_length)
    
    print("----------------------------------------------------------------------")
    print("TRAIN RNN")
    print("----------------------------------------------------------------------")
 
    rnn_nn = nn.NeuralNetwork(rnn.RNN(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]))
    rnn_data = data
    rnn_errors_train, rnn_errors_tst = rnn_nn.run_with_cross_validation(rnn_data, cross_validation_passes)
 
    print("----------------------------------------------------------------------")
    print("RNN AVERAGE ERROR = {0}".format(np.average(rnn_errors_tst)))
    print("RNN TEST ERRORS = {0}".format(rnn_errors_tst))
    print("----------------------------------------------------------------------")
    print()
    print()
    print("----------------------------------------------------------------------")
    print("TRAIN MLP")
    print("----------------------------------------------------------------------")

    mlp_nn = nn.NeuralNetwork(perceptron.MLP(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]))
    mlp_data = data
    mlp_errors_train, mlp_errors_tst = mlp_nn.run_with_cross_validation(mlp_data, cross_validation_passes)

    print("----------------------------------------------------------------------")
    print("MLP AVERAGE ERROR = {0}".format(np.average(mlp_errors_tst)))
    print("MLP TEST ERRORS = {0}".format(mlp_errors_tst))
    print("----------------------------------------------------------------------")

    perf = cp.ClassificationPerformance()
    perf.add("RNN", rnn_errors_tst)
    perf.add("MLP", mlp_errors_tst)
    perf.compare()
    perf.make_plots()


if __name__ == '__main__':
    # main()
    # fs()
    plot_features()
