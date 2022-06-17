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
(defun σ (z)
  "Sigmoid activation function."
  (declare (type single-float z))
  (/ (1+ (exp (- z)))))

(defclass sigmoid (hidden-layer-activation
                   output-layer-activation)
  ()
  (:documentation "Sigmoid activation function."))

(defmethod activate (vector (activation sigmoid))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'σ vector))

(defmethod |activate'| (vector (activation sigmoid))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (let ((σ (σ z)))
       (* σ (- 1.0 σ))))
   vector))

;; TANH
(defclass tanh% (hidden-layer-activation
                 output-layer-activation)
  ()
  (:documentation "Hyberbolic tangent activation function."))

(defmethod activate (vector (activation tanh%))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'tanh vector))

(defmethod |activate'| (vector (activation tanh%))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (declare (type single-float z))
     (let ((t% (tanh z)))
       (* (1+ t%) (- 1.0 t%))))
   vector))

;; Leaky ReLU
(defclass leaky-relu (hidden-layer-activation)
  ((coeff :initarg  :coeff
          :initform 0f0
          :reader   leaky-relu-coeff))
  (:documentation "Leaky ReLU activation function. It returns its
argument when it is greater than zero or the argument multiplied by
@c(coeff) otherwise."))

(defmethod activate (vector (activation leaky-relu))
  (declare (type magicl:matrix/single-float vector))
  (let ((coeff (leaky-relu-coeff activation)))
    (magicl:map
     (lambda (z)
       (declare (type single-float z))
       (if (> z 0) z (* z coeff)))
     vector)))

(defmethod |activate'| (vector (activation leaky-relu))
  (declare (type magicl:matrix/single-float vector))
  (let ((coeff (leaky-relu-coeff activation)))
    (magicl:map
     (lambda (z)
       (declare (type single-float z))
       (if (> z 0) 1f0 coeff))
     vector)))

;; Softmax
(defclass softmax (output-layer-activation)
  ()
  (:documentation "Softmax activation function."))

(defmethod activate (vector (activation softmax))
  (declare (type magicl:matrix/single-float vector))
  (let ((v% (magicl:map #'exp vector)))
    (magicl:scale v% (/ (the single-float (sasum v%))))))

;; Identity
(defclass identity% (output-layer-activation)
  ()
  (:documentation "Identity activation function (does nothing on its
input)."))

(defmethod activate (vector (activation identity%))
  vector)
