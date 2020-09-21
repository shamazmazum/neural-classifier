(defpackage neural-classifier-digits
  (:use #:cl #:neural-classifier #:nibbles #:snakes)
  (:export #:make-digits-classifier
           #:train-epochs
           #:load-mnist-database
           #:*mnist-dataset-path*))
