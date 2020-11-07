;; These methods optimize matrix operation functions from magicl using BLAS.

(in-package :magicl)
;; Scalar - matrix multiplication
(defmethod .* ((source1 single-float)
               (source2 matrix/single-float)
               &optional target)
  (let ((copy (or target (deep-copy-tensor source2))))
    (magicl.blas-cffi:%sscal
     (size source2)
     source1
     (storage copy)
     1)
    copy))

(defmethod ./ ((source1 matrix/single-float)
               (source2 single-float)
               &optional target)
  (let ((copy (or target (deep-copy-tensor source1))))
    (magicl.blas-cffi:%sscal
     (size source1)
     (/ source2)
     (storage copy)
     1)
    copy))

;; Matrix addition
(defmethod .+ ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (or target (deep-copy-tensor source2))))
    (magicl.blas-cffi:%saxpy
     (size source2)
     1f0
     (storage source1) 1
     (storage copy) 1)
    copy))

(defmethod .- ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((copy (or target (deep-copy-tensor source1))))
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

(defgeneric sasum (tensor)
  (:documentation "Sum of absolute values"))

(defmethod sasum ((matrix matrix/single-float))
  (magicl.blas-cffi:%sasum
   (size matrix)
   (storage matrix)
   1))

(export '(sasum))
