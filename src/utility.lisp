(in-package :neural-classifier)

(defun nrandom-generator (&key (sigma 1d0) (mean 0d0))
  "Create a generator which generates normally(0,1) distributed values"
  (declare (type double-float sigma mean))
  (let (acc)
    (lambda ()
      (when (null acc)
        (let* ((u1 (random 1d0))
               (u2 (random 1d0))
               (n1 (* (sqrt (* -2d0 (log u1)))
                      (cos (* 2d0 pi u2))))
               (n2 (* (sqrt (* -2d0 (log u1)))
                      (sin (* 2d0 pi u2)))))
          (push n1 acc)
          (push n2 acc)))
      (let ((n (pop acc)))
        (+ mean (* sigma n))))))

(defun sigma (z)
  "Function used to calculate the output from a neuron"
  (declare (optimize (speed 3))
           (type double-float z))
  (/ 1d0 (1+ (exp (- z)))))

(defun sigma% (z)
  "The first derivative of SIGMA."
  (declare (optimize (speed 3))
           (type double-float z))
  (let ((sigma (sigma z)))
    (* sigma (- 1d0 sigma))))

(defun multiply-by-scalar (x matrix)
  "Multiply a matrix by scalar X. Seems to be missing in magicl."
  (declare (optimize (speed 3))
           (type double-float x))
  (magicl:map
   (lambda (y)
     (declare (type double-float y))
     (* x y))
   matrix))
