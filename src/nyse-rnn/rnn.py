from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
from nyse import *


def prepare_data(input_length):
    book = getTestData()
    x, y = book.getXY()

    x_temp = []
    y_temp = []
    for i in range(len(x)-input_length):
        x_temp.append(x[i:(i+input_length)])
        y_temp.append(1 if y[i+input_length-1] < y[i+input_length] else 0)

    x = np.array(x_temp)
    y = np.array(y_temp)
    return x, y


def prepare_model(input_length, hidden_cnt):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(output_dim=hidden_cnt, input_dim=4, input_length=input_length, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(hidden_cnt, activation='sigmoid'))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    # try using different optimizers and different optimizer configs
    print('Compile model...')
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode="binary")
    return model


def main():

    input_length = 100
    hidden_cnt = 50

    x_train, y_train = prepare_data(input_length);
    model = prepare_model(input_length, hidden_cnt)

    # print(x_train)
    # print(y_train)
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    print("Train...")
    model.fit(x_train, y_train, validation_split=0.1, batch_size=32, nb_epoch=10, show_accuracy=True)
    y_pred = model.predict(x_train)
    print(y_pred.flatten())

if __name__ == '__main__':
    main()