;; These methods optimize matrix operation functions from magicl using BLAS.

(in-package :magicl)
(defun copy-matrix (source &optional target)
  (declare (optimize (speed 3))
           (type matrix/single-float source)
           (type (or matrix/single-float null) target))
  (cond
    ((eq source target)
     target)
    (target
     (let ((storage-t (storage target))
           (storage-s (storage source)))
       (declare (type (simple-array single-float) storage-s storage-t))
       (map-into storage-t #'identity storage-s)
       target))
    (t
      (deep-copy-tensor source))))

;; Scalar - matrix multiplication
(defmethod .* ((source1 single-float)
               (source2 matrix/single-float)
               &optional target)
  (let ((copy (copy-matrix source2 target)))
    (magicl.blas-cffi:%sscal
     (size source2)
     source1
     (storage copy)
     1)
    copy))

;; Scalar - matrix division
(defmethod ./ ((source1 matrix/single-float)
               (source2 single-float)
               &optional target)
  (let ((copy (copy-matrix source1 target)))
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
  (let ((source1 (if (eq target source1)
                     (deep-copy-tensor source1)
                     source1))
        (source2 (copy-matrix source2 target)))
    (magicl.blas-cffi:%saxpy
     (size source2)
     1f0
     (storage source1) 1
     (storage source2) 1)
    source2))

;; Matrix subtraction
(defmethod .- ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((source2 (if (eq target source2)
                     (deep-copy-tensor source2)
                     source2))
        (source1 (copy-matrix  source1 target)))
    (magicl.blas-cffi:%saxpy
     (size source2)
     -1f0
     (storage source2) 1
     (storage source1) 1)
    source1))

;; Matrix element-wise multiplication
(defmethod .* ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (declare (optimize (speed 3)))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((target (or target (zeros (shape source1)
                                  :type 'single-float))))
    (let ((target-st  (storage target))
          (source1-st (storage source1))
          (source2-st (storage source2)))
      (declare (type (simple-array single-float)
                     target-st source1-st source2-st))
      (map-into target-st #'* source1-st source2-st))
    target))

;; Matrix element-wise division
(defmethod ./ ((source1 matrix/single-float)
               (source2 matrix/single-float)
               &optional target)
  (declare (optimize (speed 3)))
  (policy-cond:with-expectations (> speed safety)
      ((assertion (equalp (shape source1)
                          (shape source2)))))
  (let ((target (or target (zeros (shape source1)
                                  :type 'single-float))))
    (let ((target-st  (storage target))
          (source1-st (storage source1))
          (source2-st (storage source2)))
      (declare (type (simple-array single-float)
                     target-st source1-st source2-st))
      (map-into target-st #'/ source1-st source2-st))
    target))

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

(define-compiler-macro .+ (&whole form source1 source2 &optional target)
  (declare (ignore source2))
  (when (eq source1 target)
    (warn "Inefficient use of .+: ~a" form))
  form)

(define-compiler-macro .- (&whole form source1 source2 &optional target)
  (declare (ignore source1))
  (when (eq source2 target)
    (warn "Inefficient use of .+: ~a" form))
  form)
