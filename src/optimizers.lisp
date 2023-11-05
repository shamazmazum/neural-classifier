(in-package :neural-classifier)

(sera:defconstructor memo
  (weights list)
  (biases  list))

(sera:-> make-memo (neural-network)
         (values memo &optional))
(defun make-memo (network)
  (memo
   (loop for m in (neural-network-weights network)
         collect (magicl:const
                  0f0 (magicl:shape m)
                  :type 'single-float))
   (loop for m in (neural-network-biases network)
         collect (magicl:const
                  0f0 (magicl:shape m)
                  :type 'single-float))))

(defclass optimizer ()
  ((learning-rate  :type          single-float
                   :reader        optimizer-learning-rate
                   :initarg       :learning-rate
                   :initform      *learning-rate*
                   :documentation "Learning rate of the
optimizer. Must be a small positive value.")
   (minibatch-size :type          positive-fixnum
                   :reader        optimizer-minibatch-size
                   :initarg       :minibatch-size
                   :initform      *minibatch-size*
                   :documentation "Minibatch size hyperparameter used
for learing. An integer in the range 10-100 is good.")
   (decay-rate     :type          single-float
                   :reader        optimizer-decay-rate
                   :initarg       :decay-rate
                   :initform      *decay-rate*
                   :documentation "A parameter used for L²
regularization. 0.0 is no regularization. Good values are 1-10 divided
by the dataset size."))
  (:documentation "Generic optimizer class. Not to be instantiated"))

(defclass sgd-optimizer (optimizer) ()
  (:documentation "The simplest SGD optimizer"))

(defclass memoizing-optimizer (optimizer)
  ((weights       :type list
                  :accessor optimizer-weights)
   (biases        :type list
                  :accessor optimizer-biases))
  (:documentation "Optimizer which memoizes some old state related to
weights and biases. Not to be instantiated."))

(defclass momentum-memo-optimizer (optimizer)
  ((momentum-memo  :type          memo
                   :accessor      optimizer-momentum-memo)
   (momentum-coeff :type          single-float
                   :reader        optimizer-momentum-coeff
                   :initarg       :momentum-coeff
                   :initform      *momentum-coeff*
                   :documentation "Coefficient responsible for momentum decay"))
  (:documentation "Optimizer based on momentum. Not to be instantiated."))

(defmethod initialize-instance :after ((optimizer momentum-memo-optimizer)
                                       &rest initargs
                                       &key &allow-other-keys)
  (let ((network (getf initargs :neural-network)))
    (when (not network)
      (error "Specify a network to train"))
    (setf (optimizer-momentum-memo optimizer)
          (make-memo network))))

(defclass rate-memo-optimizer (optimizer)
  ((rate-memo  :type          memo
               :accessor      optimizer-rate-memo)
   (rate-coeff :type          single-float
               :reader        optimizer-rate-coeff
               :initarg       :rate-coeff
               :initform      *learning-rate-increase-coeff*
               :documentation "Coefficient responsible to increase in learning rate"))
  (:documentation "Optimizer based on adaptive learning rate. Not to be instantiated."))

(defmethod initialize-instance :after ((optimizer rate-memo-optimizer)
                                       &rest initargs
                                       &key &allow-other-keys)
  (let ((network (getf initargs :neural-network)))
    (when (not network)
      (error "Specify a network to train"))
    (setf (optimizer-rate-memo optimizer)
          (make-memo network))))

(defclass momentum-optimizer (momentum-memo-optimizer)
  ()
  (:documentation "SGD optimizer with momentum"))

(defclass nesterov-optimizer (momentum-memo-optimizer)
  ()
  (:documentation "Nesterov accelerated SGD, improvement of SGD with
momentum"))

(defclass adagrad-optimizer (rate-memo-optimizer)
  ()
  (:documentation "Adagrad optimizer"))

(defclass rmsprop-optimizer (rate-memo-optimizer)
  ()
  (:documentation "RMSprop optimizer"))

(defclass adam-optimizer (momentum-memo-optimizer rate-memo-optimizer)
  ((corrected-momentum-coeff :type     single-float
                             :initform 1.0
                             :accessor optimizer-corrected-momentum-coeff)
   (corrected-rate-coeff     :type     single-float
                             :initform 1.0
                             :accessor optimizer-corrected-rate-coeff))
  (:documentation "ADAM optimizer"))

(defgeneric learn (optimizer neural-network samples)
  (:documentation "Update network parameters using SAMPLES for training."))

(defmethod learn ((optimizer sgd-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate (optimizer-learning-rate optimizer))
        (decay-rate    (optimizer-decay-rate    optimizer)))
    (flet ((update (x delta-x)
             (magicl:.- x (magicl:scale delta-x learning-rate) x)))
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples decay-rate)
        (let ((weights (neural-network-weights neural-network))
              (biases  (neural-network-biases  neural-network)))
          (mapc #'update weights delta-weight)
          (mapc #'update biases  delta-bias)))))
  (values))

(defmethod learn ((optimizer momentum-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate (optimizer-learning-rate optimizer))
        (decay-rate    (optimizer-decay-rate    optimizer)))
    (flet ((update (x delta-x accumulated-x)
             (magicl:.+ (magicl:scale delta-x learning-rate)
                        (magicl:scale accumulated-x
                                      (optimizer-momentum-coeff optimizer))
                        accumulated-x)
             (magicl:.- x accumulated-x x)))
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples decay-rate)
        (let ((weights (neural-network-weights neural-network))
              (biases  (neural-network-biases  neural-network))
              (memo (optimizer-momentum-memo optimizer)))
          (mapc #'update weights delta-weight (memo-weights memo))
          (mapc #'update biases  delta-bias   (memo-biases  memo))))))
  (values))

(defmethod learn ((optimizer nesterov-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate (optimizer-learning-rate optimizer))
        (decay-rate    (optimizer-decay-rate    optimizer)))
    (flet ((step1 (x accumulated-x)
             ;; Change weights & biases by accumulated value.
             ;; (Predict values for weights & biases).
             (magicl:.- x (magicl:scale accumulated-x
                                        (optimizer-momentum-coeff optimizer))
                        x))
           (step2 (x delta-x accumulated-x)
             ;; Update accumulated value.
             ;; Set corrected values for weights and biases
             (magicl:.+ (magicl:scale accumulated-x
                                      (optimizer-momentum-coeff optimizer))
                        (magicl:scale delta-x learning-rate)
                        accumulated-x)
             (magicl:.- x accumulated-x x)))
      (let* ((weights (neural-network-weights neural-network))
             (biases  (neural-network-biases neural-network))
             (weights-copy (mapcar #'magicl::deep-copy-tensor weights))
             (biases-copy  (mapcar #'magicl::deep-copy-tensor biases))
             (memo (optimizer-momentum-memo optimizer))
             (acc-weights (memo-weights memo))
             (acc-biases  (memo-biases  memo)))
        ;; Predict weights & biases
        (mapc #'step1 weights acc-weights)
        (mapc #'step1 biases  acc-biases)
        ;; Calculate gradient at predicted point
        (multiple-value-bind (delta-weight delta-bias)
            (calculate-gradient-minibatch neural-network samples decay-rate)
          ;; Calculate the final position
          (mapc #'step2 weights-copy delta-weight acc-weights)
          (mapc #'step2 biases-copy  delta-bias   acc-biases))

        ;; Set the new weights & biases
        (setf (neural-network-weights neural-network) weights-copy
              (neural-network-biases  neural-network) biases-copy))))
  (values))

(defmethod learn ((optimizer adagrad-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate (optimizer-learning-rate optimizer))
        (decay-rate    (optimizer-decay-rate    optimizer)))
    (flet ((update (x delta-x accumulated-x)
             (magicl:.+
              (magicl:.* delta-x delta-x)
              accumulated-x
              accumulated-x)
             (magicl:.-
              x
              (magicl:./
               (magicl:scale delta-x learning-rate)
               (magicl:map #'sqrt (magicl:.+ accumulated-x 1f-12)))
              x)))
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples decay-rate)
        (let ((weights (neural-network-weights neural-network))
              (biases  (neural-network-biases  neural-network))
              (memo    (optimizer-rate-memo    optimizer)))
          (mapc #'update weights delta-weight (memo-weights memo))
          (mapc #'update biases  delta-bias   (memo-biases  memo))))))
  (values))

(defmethod learn ((optimizer rmsprop-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate (optimizer-learning-rate optimizer))
        (decay-rate    (optimizer-decay-rate    optimizer))
        (rate-coeff    (optimizer-rate-coeff    optimizer)))
    (declare (type single-float rate-coeff))
    (flet ((update (x delta-x accumulated-x)
             (magicl:.+
              (magicl:scale (magicl:.* delta-x delta-x) (- 1.0 rate-coeff))
              (magicl:scale accumulated-x rate-coeff)
              accumulated-x)
             (magicl:.-
              x
              (magicl:./
               (magicl:scale delta-x learning-rate)
               (magicl:map #'sqrt (magicl:.+ accumulated-x 1f-12)))
              x)))
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples decay-rate)
        (let ((weights (neural-network-weights neural-network))
              (biases  (neural-network-biases  neural-network))
              (memo    (optimizer-rate-memo    optimizer)))
          (mapc #'update weights delta-weight (memo-weights memo))
          (mapc #'update biases  delta-bias   (memo-biases  memo))))))
  (values))

(defmethod learn ((optimizer adam-optimizer) neural-network samples)
  #.(declare-optimizations)
  (let ((learning-rate  (optimizer-learning-rate  optimizer))
        (decay-rate     (optimizer-decay-rate     optimizer))
        (rate-coeff     (optimizer-rate-coeff     optimizer))
        (momentum-coeff (optimizer-momentum-coeff optimizer)))
    (declare (type single-float rate-coeff momentum-coeff))
    (with-accessors ((corrected-rate-coeff     optimizer-corrected-rate-coeff)
                     (corrected-momentum-coeff optimizer-corrected-momentum-coeff))
        optimizer
      (declare (type single-float corrected-rate-coeff corrected-momentum-coeff))
      (setf corrected-rate-coeff (* corrected-rate-coeff rate-coeff)
            corrected-momentum-coeff (* corrected-momentum-coeff momentum-coeff))
      (flet ((update (x delta-x accumulated-rate accumulated-momentum)
               (magicl:.+
                (magicl:scale (magicl:.* delta-x delta-x) (- 1 rate-coeff))
                (magicl:scale accumulated-rate rate-coeff)
                accumulated-rate)
               (magicl:.+
                (magicl:scale delta-x (- 1 momentum-coeff))
                (magicl:scale accumulated-momentum momentum-coeff)
                accumulated-momentum)
               (let ((corrected-rate (magicl:scale accumulated-rate
                                                   (/ (- 1 corrected-rate-coeff))))
                     (corrected-momentum (magicl:scale accumulated-momentum
                                                       (/ (- 1 corrected-momentum-coeff)))))
                 (magicl:.-
                  x
                  (magicl:./
                   (magicl:scale corrected-momentum learning-rate)
                   (magicl:map #'sqrt (magicl:.+ corrected-rate 1f-12)))
                  x))))
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples decay-rate)
        (let ((weights (neural-network-weights neural-network))
              (biases  (neural-network-biases  neural-network))
              (momentum-memo (optimizer-momentum-memo  optimizer))
              (rate-memo     (optimizer-rate-memo      optimizer)))
          (mapc #'update weights delta-weight
                (memo-weights rate-memo) (memo-weights momentum-memo))
          (mapc #'update biases delta-bias
                (memo-biases rate-memo) (memo-biases momentum-memo)))))))
  (values))
