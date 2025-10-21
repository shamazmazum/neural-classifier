(in-package :neural-classifier)

(sera:-> standard-random ()
         (values single-float &optional))
(defun standard-random ()
  "Return a random value sampled from a distribution N(0, 1)."
  #.(declare-optimizations)
  (let ((u1 (random 1f0))
        (u2 (random 1f0)))
    (if (zerop u1)
        (standard-random)
        (* (sqrt (* -2.0 (log u1)))
           (cos (* 2 (float pi 0f0) u2))))))

(sera:-> nrandom-generator (&key (:μ single-float) (:σ single-float))
         (values (sera:-> () (values single-float &optional)) &optional))
(defun nrandom-generator (&key (μ 0f0) (σ 1f0))
  "Return a function which generates random values from a distibution
N(μ, σ)."
  #.(declare-optimizations)
  (lambda ()
    (+ μ (* σ (standard-random)))))

(sera:-> idx-abs-max (magicl:matrix/single-float)
         (values non-negative-fixnum &optional))
(defun idx-abs-max (matrix)
  "Returns index of first element with maximal absolute value by
  calling isamax() function from BLAS. Works only for rows or
  columns."
  #.(declare-optimizations)
  (let ((shape (magicl:shape matrix)))
    (if (notany
         (lambda (x)
           (declare (type fixnum x))
           (= x 1))
         shape)
        (error "This matrix is not a row or column.")))
  (let* ((storage (magicl::storage matrix))
         (max (reduce #'max storage :key #'abs)))
    (declare (type (simple-array single-float (*)) storage))
    (position max storage)))
