# StockPredictionRNN
High Frequency Trading Price Prediction using LSTM Recursive Neural Networks   

In this project we try to use recurrent neural network with long short term memory 
to predict prices in high frequency stock exchange. This program implements such a solution 
on data from NYSE OpenBook history which allows to recreate the limit order book for any given time.
Everything is described in our paper: [project.pdf](https://github.com/dzitkowskik/StockPredictionRNN/blob/master/docs/project.pdf)

### Project done for course of Computational Intelligence in Business Applications at Warsaw University of Technology - Department of Mathematics and Computer Science
http://pages.mini.pw.edu.pl/~stokowiecw/CIBA/slides/lab2.pdf   

# Installation and usage

Program is written in Python 2.7 with usage of library [Keras](http://keras.io) - [installation instruction](http://keras.io/#installation)
To install it one may need Theano installed as well as numpy, scipy, pyyaml, HDF5, h5py, cuDNN (not all are actually needed).
It is useful to install also OpenBlas.

```bash
sudo pip install git+git://github.com/Theano/Theano.git
sudo pip install keras
```

We use numpy, scipy, matplotlib and pymongo in this project so it will be useful to have them installed.

```bash
sudo pip install numpy scipy matplotlib pymongo
```

To run the program (creating folder symbols is necessary):

```bash
cd StockPredictionRNN
cd src/nyse-rnn
mkdir symbols
python main.py
```

To save data to mongodb one has to install it first [mongo install](https://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/)

Look into the code, it may be necessary to uncomment some lines to enable different features.

# Performance 

To use CUDA and OpenBlas create file ~/.theanorc and fill it with this content:

```
[global]
floatX = float32 
device = gpu1

[blas]
ldflags = −L/usr/local/lib −lopenblas

[nvcc]
fastmath = True
```

# Data



# License 

The MIT License (MIT)

Copyright (c) 2016 Karol Dzitkowski

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
