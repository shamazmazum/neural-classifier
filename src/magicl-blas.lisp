;; These methods optimize matrix operation functions from magicl using BLAS.

(in-package :magicl)
;; Scalar - matrix multiplication
(defmethod .* ((source1 double-float)
               (source2 matrix/double-float)
               &optional target)
  (declare (ignore target))
  (let ((copy (deep-copy-tensor source2)))
    (magicl.blas-cffi:%dscal
     (size source2)
     source1
     (storage copy)
     1)
    copy))

;; Matrix addition
(defmethod .+ ((source1 matrix/double-float)
               (source2 matrix/double-float)
               &optional target)
  (declare (ignore target))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (deep-copy-tensor source2)))
    (magicl.blas-cffi:%daxpy
     (size source2)
     1d0
     (storage source1) 1
     (storage copy) 1)
    copy))

(defmethod .- ((source1 matrix/double-float)
               (source2 matrix/double-float)
               &optional target)
  (declare (ignore target))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (deep-copy-tensor source1)))
    (magicl.blas-cffi:%daxpy
     (size source2)
     -1d0
     (storage source2) 1
     (storage copy) 1)
    copy))

(defmethod map! ((function function) (tensor matrix))
  (map-into
   (storage tensor)
   function
   (storage tensor))
  tensor)
