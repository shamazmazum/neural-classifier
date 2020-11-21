# Neural-classifier
[![Build Status](https://api.cirrus-ci.com/github/shamazmazum/neural-classifier.svg)](https://cirrus-ci.com/github/shamazmazum/neural-classifier)

`neural-classifier` is a neural network library based on the first chapters
from [this book](http://neuralnetworksanddeeplearning.com/). It is divided on
two systems: `neural-classifier` which is a general API for neural networks
and `neural-classifier/mnist` which contains helper functions for working with
MNIST/EMNIST datasets. For API documentation visit
[this page](http://shamazmazum.github.io/neural-classifier).

## How to work with MNIST dataset?

* Create a directory and place MNIST data in it. There are 4 files in the MNIST
  set: `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`,
  `train-images-idx3-ubyte` and ` train-labels-idx1-ubyte`.
* Load `neural-classifier/mnist` system: `(ql:quickload
  :neural-classifier/mnist)`.
* Set `neural-classifier-mnist:*mnist-dataset-path*` to your directory with
  MNIST data and execute `(neural-classifier-mnist:load-mnist-database)` (this
  will take about 10-15 seconds).
* Create a neural network: `(defparameter *nn*
  neural-classifier-mnist:make-mnist-classifier 35)` where `35` is a number of
  hidden neurons.
* Execute `(neural-classifier-mnist:train-epochs *nn* 10)` to train the network
  for 10 epochs. This function will return data about the network's accuracy for
  each epoch.
* To test your own digits convert them to `784x1` matrix of type
  `magicl:matrix/single-float` and pass it to `neural-classifier:calculate`
  function.
* Also you can play with some other hyper-parameters, not only the number of
  hidden neurons. `neural-classifier:*learn-rate*` is how fast gradient descent
  algorithm works (i.e. how fast your network learns),
  `neural-classifier:*decay-rate*` is related to regularization and should be
  about `5/N` where `N` is a number of training samples. Zero means no
  regularization.

## Dependencies

* `blas` and `lapack` foreign libraries.
* `magicl` for matrix operations.
* `nibbles` for loading MNIST data.

`magicl` and `nibbles` can be downloaded with `quicklisp`.

## What if the network shows good accuracy but fails to recognize my own digits?

If the accuracy returned by `train-epochs` is good, but the network fails to
recognize digits draws by your own hand, try EMNIST database instead of
MNIST. Copy four `emnist-digits-*` files to your MNIST directory **preserving
the name of destination files**. Images in EMNIST set are transposed (x and y
coordinates swapped), so do the same with your own images.
