(in-package :neural-classifier)

;; Variety of activation fnctions
(defgeneric activate (vector activation)
  (:documentation "Apply activation function ACTIVATION to a
VECTOR. VECTOR is an output vector from a layer of a neural network."))

(defgeneric |activate'| (vector type)
  (:documentation "Apply derivative of activation function ACTIVATION
to a VECTOR. VECTOR is an output vector from a layer of a neural
network."))

;; Activation functions are represented as classes
(defclass activation () ()
  (:documentation "Generic class for activation functions. Not to be
instantiated."))

(defclass hidden-layer-activation (activation) ()
  (:documentation "Generic class for activation functions associated
with hidden layers. Not to be instantiated."))

(defclass output-layer-activation (activation) ()
  (:documentation "Generic class for activation functions associated
with an output layer. Not to be instantiated."))

;; Sigmoid
(sera:-> σ (single-float)
         (values (single-float 0.0 1.0) &optional))
(defun σ (z)
  "Sigmoid activation function"
  #.(declare-optimizations)
  (/ (1+ (exp (- z)))))

(defclass sigmoid (hidden-layer-activation
                   output-layer-activation)
  ()
  (:documentation "Sigmoid activation function:
\\(f(x) = \\frac{1}{1 + \\exp(-x)}\\)

Has output in the range \\([0, 1]\\), so it's most suited for
describing 'intensity' of some property."))

(defmethod activate (vector (activation sigmoid))
  #.(declare-optimizations)
  (magicl:map #'σ vector))

(defmethod |activate'| (vector (activation sigmoid))
  #.(declare-optimizations)
  (magicl:map
   (lambda (z)
     (let ((σ (σ z)))
       (* σ (- 1.0 σ))))
   vector))

;; TANH
(defclass %tanh (hidden-layer-activation
                 output-layer-activation)
  ()
  (:documentation "Hyberbolic tangent activation function. Has output
in the range \\([-1, 1]\\), so it's a rescaled sigmoid. Neural
networks which use tanh in place of sigmoid are believed to be more
trainable."))

(defmethod activate (vector (activation %tanh))
  #.(declare-optimizations)
  (magicl:map #'tanh vector))

(defmethod |activate'| (vector (activation %tanh))
  #.(declare-optimizations)
  (magicl:map
   (lambda (z)
     (declare (type single-float z))
     (let ((%t (tanh z)))
       (* (1+ %t) (- 1.0 %t))))
   vector))

;; Leaky ReLU
(defclass leaky-relu (hidden-layer-activation)
  ((coeff :initarg       :coeff
          :initform      0f0
          :type          single-float
          :reader        leaky-relu-coeff
          :documentation "Coefficient of leaky ReLU. A value of 0
means just an ordinary ReLU."))
  (:documentation "Leaky ReLU activation function. It returns its
argument when it is greater than zero or the argument multiplied by
@c(coeff) otherwise. Usually this is an activation function of choice
for hidden layers."))

(defmethod activate (vector (activation leaky-relu))
  #.(declare-optimizations)
  (let ((coeff (leaky-relu-coeff activation)))
    (declare (type single-float coeff))
    (magicl:map
     (lambda (z)
       (declare (type single-float z))
       (if (> z 0) z (* z coeff)))
     vector)))

(defmethod |activate'| (vector (activation leaky-relu))
  #.(declare-optimizations)
  (let ((coeff (leaky-relu-coeff activation)))
    (declare (type single-float coeff))
    (magicl:map
     (lambda (z)
       (declare (type single-float z))
       (if (> z 0) 1f0 coeff))
     vector)))

;; Softmax
(defclass softmax (output-layer-activation)
  ()
  (:documentation "Softmax activation function: \\(f(x_i) =
\\frac{\\exp(x_i)}{\\sum_i \\exp(x_i)}\\).
It's output range is \\([0, 1]\\) and a sum of all elements in the
output vector is 1."))

(defmethod activate (vector (activation softmax))
  #.(declare-optimizations)
  (let* ((v (magicl:map #'exp vector))
         (sum (magicl:sum v)))
    (declare (type single-float sum))
    (magicl:scale v (/ sum))))

;; Identity
(defclass %identity (output-layer-activation)
  ()
  (:documentation "Identity activation function (just returns its input)."))

(defmethod activate (vector (activation %identity))
  vector)
