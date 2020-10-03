;; These methods optimize matrix operation functions from magicl using BLAS.

(in-package :magicl)
;; Scalar - matrix multiplication
(defmethod .* ((source1 single-float)
               (source2 matrix/single-float)
               &optional target)
  (declare (ignore target))
  (let ((copy (deep-copy-tensor source2)))
    (magicl.blas-cffi:%sscal
     (size source2)
     source1
     (storage copy)
     1)
    copy))

;; Matrix addition
(defmethod .+ ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (declare (ignore target))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (deep-copy-tensor source2)))
    (magicl.blas-cffi:%saxpy
     (size source2)
     1f0
     (storage source1) 1
     (storage copy) 1)
    copy))

(defmethod .- ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (declare (ignore target))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (deep-copy-tensor source1)))
    (magicl.blas-cffi:%saxpy
     (size source2)
     -1f0
     (storage source2) 1
     (storage copy) 1)
    copy))

(defmethod map! ((function function) (tensor matrix))
  (map-into
   (storage tensor)
   function
   (storage tensor))
  tensor)
