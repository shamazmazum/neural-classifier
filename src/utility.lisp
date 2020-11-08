(in-package :neural-classifier)

(declaim (ftype (function (activation-symbol)
                          (values function &optional))
                activation-fn activation-fn-derivative)
         (optimize (speed 3)))

(defun random-normal (&key (mean 0f0) (sigma 0f0))
  (declare (type single-float mean sigma))
  (float
   (cl-randist:random-normal
    (float mean 0d0)
    (float sigma 0d0))
   0f0))

;; Activation functions
(defun sigma (z)
  (declare (type single-float z))
  (/ (1+ (exp (- z)))))

(defun sigma-derivative (z)
  (declare (type single-float z))
  (let ((s (sigma z)))
    (* s (- 1.0 s))))

(defun tanh-derivative (z)
  (declare (type single-float z))
  (let ((t% (tanh z)))
    (* (1+ t%) (- 1.0 t%))))

(defun rlu (z)
  (declare (type single-float z))
  (abs z))

(defun rlu-derivative (z)
  (declare (type single-float z))
  (signum z))

(defun softmax (v)
  (declare (type magicl:matrix/single-float v))
  (let ((v% (magicl:map #'exp v)))
    (magicl:./ v%
               (magicl:sasum v%))))

(defun softmax-derivative (z)
  (declare (ignore z))
  (error "Why I am here?"))

(defun activation-fn (symbol)
  (declare (type activation-symbol symbol))
  (ecase symbol
    (:sigmoid (lambda (v) (magicl:map #'sigma v)))
    (:tanh    (lambda (v) (magicl:map #'tanh v)))
    (:rlu     (lambda (v) (magicl:map #'rlu v)))
    (:softmax #'softmax)))

(defun activation-fn-derivative (symbol)
  (declare (type activation-symbol symbol))
  (ecase symbol
    (:sigmoid (lambda (v) (magicl:map #'sigma-derivative v)))
    (:tanh    (lambda (v) (magicl:map #'tanh-derivative v)))
    (:rlu     (lambda (v) (magicl:map #'rlu-derivative v)))
    (:softmax #'softmax-derivative)))

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
