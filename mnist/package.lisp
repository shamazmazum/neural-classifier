(defpackage neural-classifier-mnist
  (:use #:cl #:nibbles)
  (:export #:make-mnist-classifier
           #:train-epochs
           #:load-mnist-database
           #:*mnist-dataset-path*))
