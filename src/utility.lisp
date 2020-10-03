(in-package :neural-classifier)

(defconstant +pi-single+ (float pi 0f0))

(defun nrandom-generator (&key (sigma 1f0) (mean 0f0))
  "Create a generator which generates normally(mean, sigma) distributed values"
  (declare (type single-float sigma mean))
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

(defun sigma (z)
  "Function used to calculate the output from a neuron"
  (declare (optimize (speed 3))
           (type single-float z))
  (/ (1+ (exp (- z)))))

(defun sigma% (z)
  "The first derivative of @c(sigma)."
  (declare (optimize (speed 3))
           (type single-float z))
  (let ((sigma (sigma z)))
    (* sigma (- 1f0 sigma))))
