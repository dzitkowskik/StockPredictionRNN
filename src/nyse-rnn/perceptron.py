from keras.models import Sequential
from keras.layers.core import TimeDistributedMerge, TimeDistributedDense, Dense, Dropout, Activation
from nyse import *
from nn import *
from keras.optimizers import SGD
# import theano
# theano.compile.mode.Mode(linker='py', optimizer='fast_compile')


class MLP:
    def __init__(self, input_length, hidden_cnt, input_dim=9, output_dim=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.hidden_cnt = hidden_cnt
        self.model = self.__prepare_model()

    def __prepare_model(self):
        print('Build model...')
        model = Sequential()
        model.add(TimeDistributedDense(output_dim=self.hidden_cnt,
                                       input_dim=self.input_dim,
                                       input_length=self.input_length,
                                       activation='sigmoid'))
        model.add(TimeDistributedMerge(mode='ave'))
        model.add(Dropout(0.1))
        model.add(Dense(self.hidden_cnt, activation='sigmoid'))
        model.add(Dense(self.output_dim))
        model.add(Activation('softmax'))

        # try using different optimizers and different optimizer configs
        print('Compile model...')
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        return model

    def prepare_data(self, book):
        x, y = book.getXY()
        x_temp = []
        y_temp = []
        for i in range(len(x)-self.input_length):
            x_temp.append(x[i:(i+self.input_length)])
            y_temp.append(y[i+self.input_length])

        x = np.array(x_temp)
        y = np_utils.to_categorical(y_temp, self.output_dim)

        print("{0} records with price down".format(sum(y[:, 0])))
        print("{0} records with price stable".format(sum(y[:, 1])))
        print("{0} records with price down".format(sum(y[:, 2])))
        print('x shape:', x.shape)
        print('y shape:', y.shape)

        return Data(x, y)

    def change_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.model = self.__prepare_model()

    def get_model(self):
        return self.model


def main():
    book = getTestData()

    input_length = 100
    hidden_cnt = 50
    nn = NeuralNetwork(MLP(input_length, hidden_cnt))
    data = nn.nn.prepare_data(book)
    print("TRAIN")
    nn.train(data)
    print("TEST")
    nn.test(data)
    print("TRAIN WITH CROSS-VALIDATION")
    nn.run_with_cross_validation(data, 2)
    print("FEATURE SELECTION")
    features = nn.feature_selection(data)
    print("Selected features: {0}".format(features))

if __name__ == '__main__':
    main()