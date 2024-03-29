# Neural-classifier
[![Build Status](https://api.cirrus-ci.com/github/shamazmazum/neural-classifier.svg)](https://cirrus-ci.com/github/shamazmazum/neural-classifier)
[![Tests](https://github.com/shamazmazum/neural-classifier/actions/workflows/test.yml/badge.svg)](https://github.com/shamazmazum/neural-classifier/actions/workflows/test.yml)

`neural-classifier` is a neural network library based on the first chapters
from [this book](http://neuralnetworksanddeeplearning.com/). It is divided on
two systems: `neural-classifier` which is a general API for neural networks
and `neural-classifier/mnist` which contains helper functions for working with
MNIST/EMNIST datasets. For API documentation visit
[this page](http://shamazmazum.github.io/neural-classifier).

## How to work with MNIST dataset?

* Unpack files in `mnist/dataset` directory.
* Load `neural-classifier/mnist` system: `(ql:quickload
  :neural-classifier/mnist)`.
* Eval `(neural-classifier-mnist:load-mnist-database)` (this will take about
  10-15 seconds).
* Create a neural network: `(defparameter *nn*
  neural-classifier-mnist:make-mnist-classifier 35)` where `35` is a number of
  hidden neurons.
* Execute `(neural-classifier-mnist:train-epochs *nn* 10)` to train the network
  for 10 epochs. This function will return data about the network's accuracy for
  each epoch.
* To test your own digits convert them to `784x1` matrix of type
  `magicl:matrix/single-float` and pass it to `neural-classifier:calculate`
  function.

## How to build custom nets and data?

See GH pages for this project (link above). In general you need to write
functions which translate your data and labels into `magicl:matrix/single-float`
matrices. Then you create a net with `neural-classifier:make-neural-network`
function and `snakes` generator which returns conses in the form `(DATA
. LABEL)`. To train a network for one epoch you call
`(neural-classifier:train-epoch)`.

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
