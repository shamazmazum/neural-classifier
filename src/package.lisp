(defpackage neural-classifier
  (:use #:cl)
  (:export #:neural-network
           #:neural-network-layout
           #:neural-network-input-trans
           #:neural-network-output-trans
           #:neural-network-input-trans%
           #:neural-network-label-trans

           #:make-neural-network
           #:calculate
           #:train-epoch
           #:rate

           #:idx-abs-max

           #:sgd-optimizer
           #:momentum-optimizer
           #:nesterov-optimizer
           #:adagrad-optimizer
           #:rmsprop-optimizer
           #:make-optimizer

           #:*momentum-coeff*
           #:*learn-rate*
           #:*decay-rate*
           #:*minibatch-size*))
