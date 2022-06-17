(defpackage neural-classifier-mnist
  (:use #:cl #:nibbles)
  (:local-nicknames (:alex :alexandria)
                    (:nc   :neural-classifier))
  (:export #:make-mnist-classifier
           #:train-epochs
           #:load-mnist-database
           #:*mnist-dataset-path*))
