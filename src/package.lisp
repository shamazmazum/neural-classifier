(defpackage neural-classifier
  (:use #:cl #:alexandria)
  (:local-nicknames (#:sera #:serapeum))
  (:export #:neural-network
           #:neural-network-layout
           #:neural-network-input-trans
           #:neural-network-output-trans
           #:neural-network-input-trans%
           #:neural-network-label-trans

           ;; Activation functions
           #:activation
           #:hidden-layer-activation
           #:output-layer-activation
           #:sigmoid
           #:%tanh
           #:softmax
           #:leaky-relu
           #:%identity

           #:make-neural-network
           #:calculate
           #:train-epoch
           #:rate

           #:idx-abs-max

           #:sgd-optimizer
           #:memoizing-optimizer
           #:momentum-optimizer
           #:nesterov-optimizer
           #:adagrad-optimizer
           #:rmsprop-optimizer

           #:*momentum-coeff*
           #:*learning-rate-increase-coeff*
           #:*learning-rate*
           #:*decay-rate*
           #:*minibatch-size*))
