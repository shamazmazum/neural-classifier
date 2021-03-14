(in-package :neural-classifier)

(deftype activation-symbol ()
  '(member :sigmoid :tanh :abs :relu :softmax))

(defclass neural-network ()
  ((layout           :initarg       :layout
                     :initform      (error "Specify number of neurons in each layer")
                     :type          list
                     :reader        neural-network-layout
                     :documentation "Number of neurons in each layer of the network")
   (activation-funcs :initarg       :activation-funcs
                     :initform      nil
                     :type          list
                     :accessor      neural-network-activation-funcs
                     :documentation "List of activation functions.")
   (weights          :type          list
                     :accessor      neural-network-weights
                     :documentation "Weight matrices for each layer")
   (biases           :type          list
                     :accessor      neural-network-biases
                     :documentation "Bias vectors for each layer")
   (input-trans      :type          function
                     :initarg       :input-trans
                     :initform      #'identity
                     :accessor      neural-network-input-trans
                     :documentation "Function which translates an input object to a vector")
   (output-trans     :type          function
                     :initarg       :output-trans
                     :initform      #'identity
                     :accessor      neural-network-output-trans
                     :documentation "Function which translates an output vector to a label.")
   (input-trans%     :type          function
                     :initarg       :input-trans%
                     :initform      #'identity
                     :accessor      neural-network-input-trans%
                     :documentation "Function which translates an input object to a vector
(used for training)")
   (label-trans      :type          function
                     :initarg       :label-trans
                     :initform      #'identity
                     :accessor      neural-network-label-trans
                     :documentation "Function which translates a label to a vector"))
  (:documentation "Class for neural networks"))

(deftype non-negative-fixnum () '(integer 0 #.most-positive-fixnum))
(deftype positive-fixnum () '(integer 1 #.most-positive-fixnum))

(declaim (type single-float
               *learn-rate* *decay-rate*)
         (type positive-fixnum *minibatch-size*))
(defparameter *learn-rate* 0.005f0
  "Learning speed for gradient descent algorithms. Bigger values
result in faster learning, but too big is bad. Default value is good
for SGD, SGD with momentum and NAG optimizers. For Adagrad and
RMSprop try 0.001f0.")
(defparameter *decay-rate* 0f0
  "Regularization parameter @c(λ/N), where @c(N) is the number of
objects in the training set and @c(λ) must be about 1-10. If not sure,
start with zero (which is the default).")
(defparameter *minibatch-size* 10
  "Number of samples to be used for one update of network parameters.")
(defparameter *momentum-coeff* 0.8
  "Hyperparameter for SGD optimizers which use momentum. Zero means
just usual SGD. RMSprop also uses this parameter in accumulation of
squared partial derivatives of network parameters. Good values are
0.8-0.9.")
