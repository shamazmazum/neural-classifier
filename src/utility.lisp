(in-package :neural-classifier)

(declaim (optimize (speed 3)))

(defun standard-random ()
  "Return a random value sampled from a distribution N(0, 1)."
  (let ((u1 (random 1f0))
        (u2 (random 1f0)))
    (if (zerop u1)
        (standard-random)
        (* (sqrt (* -2.0 (log u1)))
           (cos (* 2 (float pi 0f0) u2))))))

(defun nrandom-generator (&key (μ 0f0) (σ 1f0))
  "Return a function which generates random values from a distibution
N(μ, σ)."
  (declare (type single-float μ σ))
  (lambda ()
    (+ μ (* σ (standard-random)))))

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
