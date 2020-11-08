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

           #:*learn-rate*
           #:*decay-rate*
           #:*minibatch-size*))
