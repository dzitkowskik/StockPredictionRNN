import rnn
import perceptron
from nyse import *
import nn
import classification_performance as cp
from plotting import *


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
    output = open('../../results/RNN_features', 'wb')
    pickle.dump(feature_selection, output)
    output.close()


def rrn_iter_error(iters=200):

    input_length = 100
    hidden_cnt = 100
    data = get_test_data(input_length)

    print("input length ", input_length)
    print("hidden_cnt ", hidden_cnt)

    errors = {}
    errors["test"] = []
    errors["train"] = []

    n = data.x.shape[0]

    x_train = data.x[:(n/2), :]
    y_train = data.y[:(n/2), :]
    x_test = data.x[(n/2):, :]
    y_test = data.y[(n/2):, :]

    data_train = Data(x_train, y_train)
    data_test = Data(x_test, y_test)

    print('train x shape:', data_train.x.shape)
    print('train y shape:', data_train.y.shape)
    print('test x shape:', data_test.x.shape)
    print('test y shape:', data_test.y.shape)

    print("----------------------------------------------------------------------")
    print("TRAIN RNN")
    print("----------------------------------------------------------------------")

    # try to get best nn
    best_error = 1
    for x in range(5):
        rnn_nn = nn.NeuralNetwork(rnn.RNN(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]), nb_epoch=20)
        error_train = rnn_nn.train(data_train)
        if error_train < best_error:
            best_error = error_train
            best_nn = rnn_nn

    rnn_nn = best_nn
    rnn_nn.nb_epoch = 10
    for i in range(iters):
        print("-----------------------------")
        print("ITERATION {0} FROM {1}".format(i, iters))
        print("-----------------------------")

        error_train = rnn_nn.train(data_train)
        print("Train ERROR: {0}".format(error_train))
        error_tst = rnn_nn.test(data_test)
        print("Test ERROR: {0}".format(error_tst))
        errors["train"].append(error_train)
        errors["test"].append(error_tst)

    print(errors)

    output = open('../../results/RNN_errors', 'wb')
    pickle.dump(errors, output)
    output.close()


def mlp_iter_error(iters=500):

    input_length = 50
    hidden_cnt = 100
    data = get_test_data(input_length)

    print("input length ", input_length)
    print("hidden_cnt ", hidden_cnt)

    errors = {}
    errors["test"] = []
    errors["train"] = []

    n = data.x.shape[0]

    x_train = data.x[:(n/2), :]
    y_train = data.y[:(n/2), :]
    x_test = data.x[(n/2):, :]
    y_test = data.y[(n/2):, :]

    data_train = Data(x_train, y_train)
    data_test = Data(x_test, y_test)

    print('train x shape:', data_train.x.shape)
    print('train y shape:', data_train.y.shape)
    print('test x shape:', data_test.x.shape)
    print('test y shape:', data_test.y.shape)

    print("----------------------------------------------------------------------")
    print("TRAIN MLP")
    print("----------------------------------------------------------------------")

    # try to get best nn
    best_error = 1
    for x in range(4):
        rnn_nn = nn.NeuralNetwork(perceptron.MLP(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]), nb_epoch=20)
        error_train = rnn_nn.train(data_train)
        if error_train < best_error:
            best_error = error_train
            best_nn = rnn_nn

    rnn_nn = best_nn
    rnn_nn.nb_epoch = 10
    for i in range(iters):
        print("-----------------------------")
        print("ITERATION {0} FROM {1}".format(i, iters))
        print("-----------------------------")

        error_train = rnn_nn.train(data_train)
        print("Train ERROR: {0}".format(error_train))
        error_tst = rnn_nn.test(data_test)
        print("Test ERROR: {0}".format(error_tst))
        errors["train"].append(error_train)
        errors["test"].append(error_tst)

    print(errors)

    output = open('../../results/MLP_errors', 'wb')
    pickle.dump(errors, output)
    output.close()


def main():
    input_length = 25
    hidden_cnt = 100
    cross_validation_passes = 20
    epochs = 200
    data = get_test_data(input_length)
    
    print("----------------------------------------------------------------------")
    print("TRAIN RNN")
    print("----------------------------------------------------------------------")

    rnn_nn = nn.NeuralNetwork(rnn.RNN(input_length, hidden_cnt, data.x.shape[2], data.y.shape[1]))
    rnn_data = data
    rnn_nn.nb_epoch = epochs
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
    mlp_nn.nb_epoch = epochs
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
    main()
    # fs()
    # plot_features()
    # rrn_iter_error()
    # rrn_iter_error_plot()
    # mlp_iter_error()
    # mlp_iter_error_plot()
