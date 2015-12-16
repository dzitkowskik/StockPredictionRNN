from keras.models import Graph
from keras.layers.core import Dense
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
    model = Graph()
    inputs = []
    for i in range(input_length):
        model.add_input(name='input'+str(i), input_shape=(4,))
        model.add_node(Dense(4), name='dense'+str(i), input='input'+str(i))
        inputs.append('dense'+str(i))
    # print(inputs, 'inputs')
    model.add_node(Dense(hidden_cnt, activation='sigmoid'), name='denseMerge', inputs=inputs)
    model.add_node(Dense(1, activation='softmax'), name='output', input='denseMerge', create_output=True)

    print('Compile model...')
    model.compile(optimizer='adam', loss={'output': 'binary_crossentropy'})
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
    inputs = {}
    for i in range(input_length):
        inputs['input'+str(i)] = x_train[:, i, :].reshape((x_train.shape[0], x_train.shape[2]))
        # print('input shape: ', inputs['input'+str(i)].shape)
    inputs['output'] = y_train

    history = model.fit(inputs, nb_epoch=10)
    y_pred = model.predict(inputs)
    print(y_pred)

if __name__ == '__main__':
    main()