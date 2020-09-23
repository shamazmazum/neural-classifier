(defpackage neural-classifier
  (:use #:cl)
  (:export #:neural-network
           #:neural-network-layout
           #:neural-network-input-trans
           #:neural-network-output-trans
           #:neural-network-train-trans
           #:make-neural-network
           #:calculate
           #:*learn-rate*
           #:*decay-rate*
           #:*minibatch-size*
           #:train-epoch
           #:rate))
