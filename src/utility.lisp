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

(defun activation-fn (symbol)
  (declare (type activation-symbol symbol))
  (ecase symbol
    (:sigmoid #'sigma)
    (:tanh    #'tanh)
    (:rlu     #'rlu)))

(defun activation-fn-derivative (symbol)
  (declare (type activation-symbol symbol))
  (ecase symbol
    (:sigmoid #'sigma-derivative)
    (:tanh    #'tanh-derivative)
    (:rlu     #'rlu-derivative)))
