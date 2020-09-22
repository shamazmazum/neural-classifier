(in-package :neural-classifier-digits)

(defparameter *mnist-dataset-path*
  #p"~/mnist-dataset/"
  "Path to MNIST dataset")

(defvar *train-data* nil)
(defvar *test-data* nil)

(defconstant +image-magic+ 2051)
(defconstant +label-magic+ 2049)

(defun output-transform (output)
  (declare (optimize (speed 3))
           (type magicl:matrix/double-float output))
  (let* ((output% (loop for i below 10 collect (magicl:tref output i 0)))
         (max (reduce #'max output%)))
    (declare (type list output%))
    (position max output%)))

(defun train-transform (digit)
  (declare (optimize (speed 3))
           (type (integer 0 9) digit))
  (let ((vector (magicl:zeros '(10 1))))
    (setf (magicl:tref vector digit 0) 1d0)
    vector))

(defun read-labels (which)
  (let ((name (ecase which
                (:train "train-labels-idx1-ubyte")
                (:test  "t10k-labels-idx1-ubyte"))))
    (with-open-file (input (merge-pathnames name *mnist-dataset-path*)
                           :element-type '(unsigned-byte 8))
      (if (/= +label-magic+ (read-ub32/be input))
          (error "Not a MNIST dataset"))
      (loop repeat (read-ub32/be input) collect
            (read-byte input)))))

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
        (loop repeat images collect
              (magicl:from-list
               (loop repeat pixels collect
                     (/ (read-byte input) 255d0))
               (list pixels 1)))))))

(defun load-mnist-database ()
  (setq *train-data*
        (mapcar #'cons
                (read-images :train)
                (read-labels :train))
        *test-data*
        (mapcar #'cons
                (read-images :test)
                (read-labels :test)))
  t)

(defun shuffle-list (list)
  (labels ((do-shuffle (list acc)
             (if (null list)
                 acc
                 (let ((idx (random (length list))))
                   (do-shuffle
                       (append (subseq list 0 idx)
                               (subseq list (1+ idx)))
                     (cons (nth idx list) acc))))))
    (do-shuffle list nil)))

(defun add-noise (vector)
  (declare (optimize (speed 3))
           (type magicl:matrix/double-float vector))
  (magicl:map
   (lambda (x)
     (declare (type double-float x))
     (min 1d0 (+ x (random 0.1d0))))
   vector))

(defun make-digits-classifier (inner-layers)
  ;; Just hardcode the number in the input layer
  (make-neural-network (list #.(* 28 28) inner-layers 10)
                       :input-trans  #'add-noise
                       :output-trans #'output-transform
                       :train-trans  #'train-transform))

(defun train (classifier)
  (train-epoch classifier (shuffle-list *train-data*)))

(defun rate-digits (classifier list)
  (rate classifier list))

(defun train-epochs (classifier n)
  (loop repeat n collect
       (progn
         (train classifier)
         (cons (rate-digits classifier *train-data*)
               (rate-digits classifier *test-data*)))))
