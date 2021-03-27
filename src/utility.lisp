(in-package :neural-classifier)

(declaim (optimize (speed 3)))

(defun random-normal (&key (mean 0f0) (sigma 0f0))
  (declare (type single-float mean sigma))
  (float
   (cl-randist:random-normal
    (float mean 0d0)
    (float sigma 0d0))
   0f0))

(defun idx-abs-max (matrix)
  "Returns index of first element with maximal absolute value by
  calling isamax() function from BLAS. Works only for rows or
  columns."
  (declare (type magicl:matrix/single-float matrix))
  (let ((shape (magicl:shape matrix)))
    (if (notany
         (lambda (x)
           (declare (type fixnum x))
           (= x 1))
         shape)
        (error "This matrix is not a row or column.")))
  (1- ; Transform fortran index to lisp index
   (the fixnum
        (magicl.blas-cffi:%isamax
         (magicl:size matrix)
         (magicl::storage matrix)
         1))))

(defun sasum (matrix)
  (declare (type magicl:matrix/single-float matrix))
  (magicl.blas-cffi:%sasum
   (magicl:size matrix)
   (magicl::storage matrix)
   1))

;; Activation functions
(defgeneric activation (vector type)
  (:documentation "Apply activation function TYPE to a VECTOR"))

(defgeneric activation-derivative (vector type)
  (:documentation "Apply derivative of activation function TYPE to a VECTOR"))

(defun sigmoid (z)
  (declare (type single-float z))
  (/ (1+ (exp (- z)))))

(defmethod activation (vector (type (eql :sigmoid)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'sigmoid vector))

(defmethod activation-derivative (vector (type (eql :sigmoid)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (let ((s (sigmoid z)))
       (* s (- 1.0 s))))
   vector))

(defmethod activation (vector (type (eql :tanh)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'tanh vector))

(defmethod activation-derivative (vector (type (eql :tanh)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (declare (type single-float z))
     (let ((t% (tanh z)))
       (* (1+ t%) (- 1.0 t%))))
   vector))

(defmethod activation (vector (type (eql :abs)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'abs vector))

(defmethod activation-derivative (vector (type (eql :abs)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map #'signum vector))

(defmethod activation (vector (type (eql :relu)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (declare (type single-float z))
     (max 0.0 z))
   vector))

(defmethod activation-derivative (vector (type (eql :relu)))
  (declare (type magicl:matrix/single-float vector))
  (magicl:map
   (lambda (z)
     (declare (type single-float z))
     (max 0.0 (signum z)))
   vector))

(defmethod activation (vector (type (eql :softmax)))
  (declare (type magicl:matrix/single-float vector))
  (let ((v% (magicl:map #'exp vector)))
    (magicl:scale v% (/ (the single-float (sasum v%))))))
