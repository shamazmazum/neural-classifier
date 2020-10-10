(in-package :neural-classifier)

(declaim (ftype (function (activation-symbol)
                          (values function &optional))
                activation-fn activation-fn-derivative)
         (type single-float +pi-single+)
         (optimize (speed 3)))

(defconstant +pi-single+ (float pi 0f0))

(defun nrandom-generator (&key (sigma 1f0) (mean 0f0))
  "Create a generator which generates normally(mean, sigma) distributed values"
  (declare (type single-float sigma mean)
           (optimize (speed 1)))
  (let (acc)
    (lambda ()
      (when (null acc)
        (let* ((u1 (random 1f0))
               (u2 (random 1f0))
               (n1 (* (sqrt (* -2f0 (log u1)))
                      (cos (* 2f0 +pi-single+ u2))))
               (n2 (* (sqrt (* -2f0 (log u1)))
                      (sin (* 2f0 +pi-single+ u2)))))
          (push n1 acc)
          (push n2 acc)))
      (let ((n (pop acc)))
        (+ mean (* sigma n))))))

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
