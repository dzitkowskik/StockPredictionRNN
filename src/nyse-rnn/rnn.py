from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
import numpy as np
from nyse import *
from keras.optimizers import SGD


def prepare_data(input_length):
    book = getTestData()
    x, y = book.getXY()

    x_temp = []
    y_temp = []
    for i in range(len(x)-input_length):
        x_temp.append(x[i:(i+input_length)])
        y_temp.append(y[i+input_length])

    x = np.array(x_temp)
    y = np_utils.to_categorical(y_temp, 3)
    return x, y


def prepare_model(input_length, hidden_cnt):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(output_dim=hidden_cnt, input_dim=9, input_length=input_length, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(hidden_cnt, activation='sigmoid'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    print('Compile model...')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


def main():
    input_length = 100
    hidden_cnt = 50

    x_train, y_train = prepare_data(input_length)
    print("{0} records with price down".format(sum(y_train[:, 0])))
    print("{0} records with price stable".format(sum(y_train[:, 1])))
    print("{0} records with price down".format(sum(y_train[:, 2])))
    model = prepare_model(input_length, hidden_cnt)

    # print(x_train)
    # print(y_train)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    print("Train...")
    model.fit(x_train, y_train, validation_split=0.5, batch_size=128, nb_epoch=3, show_accuracy=True)
    y_pred = np_utils.probas_to_classes(model.predict(x_train))
    y_train = np_utils.probas_to_classes(y_train)
    print("PREDICTED: class 0: {0}, class 1: {1}, class 2: {2}".format(
          np.sum(np.ravel(y_pred) == 0),
          np.sum(np.ravel(y_pred) == 1),
          np.sum(np.ravel(y_pred) == 2)))
    print("ACTUAL: class 0: {0}, class 1: {1}, class 2: {2}".format(
          np.sum(np.ravel(y_train) == 0),
          np.sum(np.ravel(y_train) == 1),
          np.sum(np.ravel(y_train) == 2)))
    print("ERROR RATE: ", (np.ravel(y_pred) != np.ravel(y_train)).sum().astype(float)/y_train.shape[0])

if __name__ == '__main__':
    main()