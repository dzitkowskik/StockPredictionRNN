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

To save data to mongodb one has to install it first [mongo install](https://docs.mongodb.org/manual/tutorial/install-mongodb-on-ubuntu/)

# Data

# License 



