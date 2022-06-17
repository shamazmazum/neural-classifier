(defsystem :neural-classifier
  :name :neural-classifier
  :version "0.2"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Classification of samples based on neural network."
  :licence "2-clause BSD"
  :pathname "src/"
  :serial t
  :components ((:file "package")
               (:file "magicl-blas")
               (:file "definitions")
               (:file "utility")
               (:file "activation")
               (:file "optimizers")
               (:file "neural-network"))
  :depends-on ((:feature :single-float-tran :sbcl-single-float-tran)
               :alexandria
               :magicl/ext-blas
               :magicl/ext-lapack
               :snakes)
  :in-order-to ((test-op (load-op "neural-classifier/tests")))
  :perform (test-op (op system)
                    (declare (ignore op system))
                    (uiop:symbol-call :neural-classifier-tests '#:run-tests)))

(defsystem :neural-classifier/mnist
  :name :neural-classifier/mnist
  :version "0.2"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Recognition of handwritten digits based on MNIST/EMNIST datasets."
  :licence "2-clause BSD"
  :pathname "mnist/"
  :serial t
  :components ((:file "package")
               (:file "mnist"))
  :depends-on (:neural-classifier :nibbles))

(defsystem :neural-classifier/tests
  :name :neural-classifier/tests
  :version "0.2"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Recognition of handwritten digits based on MNIST/EMNIST datasets."
  :licence "2-clause BSD"
  :pathname "tests/"
  :serial t
  :components ((:file "package")
               (:file "tests"))
  :depends-on (:neural-classifier/mnist :fiveam))
