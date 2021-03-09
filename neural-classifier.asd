(defsystem :neural-classifier
  :name :neural-classifier
  :version "0.1"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Classification of samples based on neural network."
  :licence "2-clause BSD"
  :pathname "src/"
  :serial t
  :components ((:file "package")
               #+sbcl
               (:file "sbcl-hacks")
               (:file "magicl-blas")
               (:file "definitions")
               (:file "utility")
               (:file "optimizers")
               (:file "neural-network"))
  :depends-on (:magicl :snakes :cl-randist))

(defsystem :neural-classifier/mnist
  :name :neural-classifier/mnist
  :version "0.1"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Recognition of handwritten digits based on MNIST/EMNIST datasets."
  :licence "2-clause BSD"
  :pathname "mnist/"
  :serial t
  :components ((:file "package")
               (:file "mnist"))
  :depends-on (:neural-classifier :nibbles))
