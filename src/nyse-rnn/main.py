import rnn
import perceptron
from nyse import *
import nn
import classification_performance as cp
import numpy as np


def main():
    input_length = 100
    hidden_cnt = 50
    cross_validation_passes = 10

    data = get_test_data(input_length)
    
    print("----------------------------------------------------------------------")
    print("TRAIN RNN")
    print("----------------------------------------------------------------------")

    rnn_nn = nn.NeuralNetwork(rnn.RNN(input_length, hidden_cnt))
    rnn_data = data
    rnn_errors_train, rnn_errors_tst = rnn_nn.run_with_cross_validation(rnn_data, cross_validation_passes)

    print("----------------------------------------------------------------------")
    print("RNN AVERAGE ERROR = {0}".format(np.average(rnn_errors_tst)))
    print("----------------------------------------------------------------------")
    print()
    print()
    print("----------------------------------------------------------------------")
    print("TRAIN RNN")
    print("----------------------------------------------------------------------")

    mlp_nn = nn.NeuralNetwork(perceptron.MLP(input_length, hidden_cnt))
    mlp_data = data
    mlp_errors_train, mlp_errors_tst = mlp_nn.run_with_cross_validation(mlp_data, cross_validation_passes)

    print("----------------------------------------------------------------------")
    print("MLP AVERAGE ERROR = {0}".format(np.average(mlp_errors_tst)))
    print("----------------------------------------------------------------------")

    perf = cp.ClassificationPerformance()
    perf.add("RNN", rnn_errors_tst)
    perf.add("MLP", mlp_errors_tst)
    perf.compare()


if __name__ == '__main__':
    main()