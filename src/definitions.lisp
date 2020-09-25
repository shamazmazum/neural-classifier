(in-package :neural-classifier)

(defclass neural-network ()
  ((layout       :initarg       :layout
                 :initform      (error "Specify number of neurons in each layer")
                 :type          list
                 :reader        neural-network-layout
                 :documentation "Number of neurons in each layer of the network")
   (weights      :type          list
                 :accessor      neural-network-weights
                 :documentation "Weight matrices for each layer")
   (biases       :type          list
                 :accessor      neural-network-biases
                 :documentation "Bias vectors for each layer")
   (input-trans  :type          function
                 :initarg       :input-trans
                 :initform      #'identity
                 :accessor      neural-network-input-trans
                 :documentation "Function which translates input object to a vector")
   (output-trans :type          function
                 :initarg       :output-trans
                 :initform      #'identity
                 :accessor      neural-network-output-trans
                 :documentation "Function which translates output vector to some object.")
   (train-trans  :type          function
                 :initarg       :train-trans
                 :initform      #'identity
                 :accessor      neural-network-train-trans
                 :documentation "Function which translates expected object to output vector"))
  (:documentation "Class for neural networks"))

(deftype non-negative-fixnum () '(integer 0 #.most-positive-fixnum))
(deftype positive-fixnum () '(integer 1 #.most-positive-fixnum))

(declaim (type double-float
               *learn-rate* *decay-rate*)
         (type positive-fixnum *minibatch-size*))
(defparameter *learn-rate* 0.005d0
  "Speed of gradient descent algorithm")
(defparameter *decay-rate* 0d0
  "Speed of weights decay λ/N")
(defparameter *minibatch-size* 10
  "Number of samples to be used in stochastic gradient descent")
