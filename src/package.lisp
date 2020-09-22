(defpackage neural-classifier
  (:use #:cl)
  (:export #:neural-network
           #:neural-network-layout
           #:make-neural-network
           #:calculate
           #:*learn-rate*
           #:*decay-rate*
           #:*minibatch-size*
           #:train-epoch
           #:rate))
