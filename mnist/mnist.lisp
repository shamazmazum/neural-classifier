(in-package :neural-classifier-mnist)

(defparameter *mnist-dataset-path*
  (asdf:system-relative-pathname
   :neural-classifier/mnist
   #p"mnist/dataset/")
  "Path to MNIST dataset")

(defvar *train-data* nil)
(defvar *test-data* nil)

(defconstant +image-magic+ 2051)
(defconstant +label-magic+ 2049)

(alex:define-constant +labels+
    (make-array 10
                :initial-contents
                (loop
                      for i below 10
                      for v = (magicl:zeros '(10 1) :type 'single-float)
                      do (setf (magicl:tref v i 0) 1.0)
                      collect v))
  :test #'equalp)

(defun label-transform (digit)
  (declare (optimize (speed 3))
           (type (integer 0 9) digit))
  (svref +labels+ digit))

(defun read-labels (which)
  (let ((name (ecase which
                (:train "train-labels-idx1-ubyte")
                (:test  "t10k-labels-idx1-ubyte"))))
    (with-open-file (input (merge-pathnames name *mnist-dataset-path*)
                           :element-type '(unsigned-byte 8))
      (if (/= +label-magic+ (read-ub32/be input))
          (error "Not a MNIST dataset"))
      (let ((length (read-ub32/be input)))
        (make-array length
                    :initial-contents
                    (loop repeat length collect
                         (read-byte input)))))))

(defun read-images (which)
  (let ((name (ecase which
                (:train "train-images-idx3-ubyte")
                (:test  "t10k-images-idx3-ubyte"))))
    (with-open-file (input (merge-pathnames name *mnist-dataset-path*)
                           :element-type '(unsigned-byte 8))
      (if (/= +image-magic+ (read-ub32/be input))
          (error "Not a MNIST dataset"))
      (let* ((images  (read-ub32/be input))
             (rows    (read-ub32/be input))
             (columns (read-ub32/be input))
             (pixels  (* rows columns)))
        (make-array images
                    :initial-contents
                    (loop repeat images collect
                         (magicl:from-list
                          (loop repeat pixels collect
                               (/ (read-byte input) 255f0))
                          (list pixels 1))))))))

(defun load-mnist-database ()
  (setq *train-data*
        (map 'vector #'cons
             (read-images :train)
             (read-labels :train))
        *test-data*
        (map 'vector #'cons
             (read-images :test)
             (read-labels :test)))
  (values))

(defun shuffle-vector (vector)
  (declare (optimize (speed 3))
           (type simple-vector vector))
  (loop
     with length = (length vector)
     for i below length
     for j = (random (- length i))
     for rnd-item = (svref vector j)
     for end-item = (svref vector (- length 1 i))
     do
       (setf (svref vector (- length 1 i)) rnd-item
             (svref vector j) end-item))
  vector)

(defun add-noise (vector)
  (declare (optimize (speed 3))
           (type magicl:matrix/single-float vector))
  (flet ((clamp (val min max)
           (min (max val min) max)))
    (magicl:map
     (lambda (x)
       (declare (type single-float x))
       (clamp (+ x (random 0.2f0) -0.1f0) 0f0 1f0))
     vector)))

(defun possibly-invert (vector)
  (declare (optimize (speed 3))
           (type magicl:matrix/single-float vector))
  (if (< (random 1.0) 0.5)
      (magicl:map
       (lambda (x)
         (declare (type single-float x))
         (- 1f0 x))
       vector)
      vector))

(defun make-mnist-classifier (inner-neurons)
  "Make a neural network to classify digits from the MNIST
dataset. @c(inner-neurons) is a number of neurons in the inner layer."
  (nc:make-neural-network
   (list #.(* 28 28) inner-neurons 10)
   :input-trans%     (alex:compose #'possibly-invert #'add-noise)
   :output-trans     #'nc:idx-abs-max
   :label-trans      #'label-transform
   :activation-funcs (list (make-instance 'nc:leaky-relu :coeff 0.2)
                           (make-instance 'nc:softmax))))

(defun train-epoch (classifier optimizer)
  (nc:train-epoch
   classifier
   (snakes:sequence->generator
    (shuffle-vector *train-data*))
   :optimizer optimizer))

(defun rate (classifier vector)
  (nc:rate
   classifier
   (snakes:sequence->generator vector)))

(defun train-epochs (classifier n
                     &optional (optimizer
                                (make-instance 'nc:momentum-optimizer
                                               :neural-network classifier)))
  "Train a neural network @c(classifier) for @c(n) epochs.
Return a list of accuracy data for each epoch of training."
  (loop repeat n collect
        (progn
          (train-epoch classifier optimizer)
          (cons (rate classifier *train-data*)
                (rate classifier *test-data*)))))

(format t "Place MNIST dataset to ~a (controlled by ~a) and run ~a~%"
        *mnist-dataset-path*
        '*mnist-dataset-path*
        'load-mnist-database)
