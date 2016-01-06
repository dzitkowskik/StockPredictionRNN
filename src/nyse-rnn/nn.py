from sklearn import cross_validation
import numpy as np


class Data:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class NeuralNetwork:
    def __init__(self, model):
        self.model = model

    def train(self, data):
        self.model.fit(data.x, data.y, validation_split=0.1, batch_size=32, nb_epoch=10, show_accuracy=True)
        return self.test(data)

    def test(self, data):
        y_pred = self.model.predict(data.x)
        error = np.sum(np.ravel(y_pred) != np.ravel(data.y)) / float(len(data.x))
        return error

    def run_with_cross_validation(self, data, cross_num):
        # N - number of observations, M - number of features
        N, M = data.x.shape
        train_errors = np.zeros(cross_num)
        test_errors = np.zeros(cross_num)
        cv = cross_validation.KFold(N, cross_num, shuffle=True)

        i = 0
        for train_index, test_index in cv:
            x_train = data.x[train_index, :]
            y_train = data.y[train_index, :]
            x_test = data.x[test_index, :]
            y_test = data.y[test_index, :]

            train_data = Data(x_train, y_train)
            test_data = Data(x_test, y_test)

            train_errors[i] = self.train(train_data)
            test_errors[i] = self.test(test_data)
            i += 1

        return train_errors, test_errors

    def feature_selection(self, data):
        # N - number of observations, M - number of features
        N, M = data.x.shape

        best_err_rate = 1
        available_features = range(M)
        selected_features = []
        temp_selected_features = []

        for i in range(M):
            progress = False
            for feature in available_features:
                current_feature_set = list(selected_features).append(feature)
                train_errors, test_errors = self.run_with_cross_validation(data[:, current_feature_set], 4)
                error = np.average(test_errors)
                if error < best_err_rate:
                    best_err_rate = error
                    temp_selected_features = list(current_feature_set)
                    progress = True
            if not progress:
                break
            selected_features = temp_selected_features

        return selected_features

