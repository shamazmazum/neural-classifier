(in-package :neural-classifier)

(defclass neural-network ()
  ((layout       :initarg       :layout
                 :initform      (error "Specify number of neurons in each layer")
                 :type          list
                 :reader        neural-network-layout
                 :documentation "Number of neurons in each layer of the network")
   (weights      :type          list
                 :accessor      neural-network-weights
                 :documentation "Input weight matrices for each layer")
   (biases       :type          list
                 :accessor      neural-network-biases
                 :documentation "Bias vectors for each layer")
   (input-trans  :type          function
                 :initarg       :input-trans
                 :initform      #'identity
                 :reader        neural-network-input-trans
                 :documentation "Function which translates input object to a vector")
   (output-trans :type          function
                 :initarg       :output-trans
                 :initform      #'identity
                 :reader        neural-network-output-trans
                 :documentation "Function which translates output vector to some object.")
   (train-trans  :type          function
                 :initarg       :train-trans
                 :initform      #'identity
                 :reader        neural-network-train-trans
                 :documentation "Function which translates expected object to output vector"))
  (:documentation "Class for MPL neural network"))

(deftype non-negative-fixnum () '(integer 0 #.most-positive-fixnum))
(deftype positive-fixnum () '(integer 1 #.most-positive-fixnum))

(declaim (type double-float
               *learn-rate* *decay-rate*)
         (type positive-fixnum *minibatch-size*))
(defparameter *learn-rate* 0.005d0
  "Speed of gradient descent algorithm")
(defparameter *decay-rate* 0d0
  "Speed of parameter decay")
(defparameter *minibatch-size* 10)
