(in-package :neural-classifier)

(defclass optimizer () ()
  (:documentation "Generic optimizer class. Not to be instantiated"))

(defclass sgd-optimizer (optimizer) ()
  (:documentation "The simplest SGD optimizer"))

(defclass memoizing-optimizer (optimizer)
  ((weights       :type list
                  :accessor optimizer-weights)
   (biases        :type list
                  :accessor optimizer-biases)
   (initial-value :type single-float
                  :accessor optimizer-initial-value
                  :initarg :initial-value
                  :initform 0f0))
  (:documentation "Optimizer which memoizes some old state related to
weights and biases. Not to be instantiated."))

(defclass momentum-optimizer (memoizing-optimizer)
  ((coeff   :type single-float
            :accessor momentum-coeff
            :initarg :coeff
            :initform *momentum-coeff*))
  (:documentation "SGD optimizer with momentum"))

(defclass nesterov-optimizer (momentum-optimizer)
  ()
  (:documentation "Nesterov accelerated SGD, improvement of SGD with
momentum"))

(defclass adagrad-optimizer (memoizing-optimizer)
  ()
  (:default-initargs
   :initial-value 1f-8)
  (:documentation "Adagrad optimizer"))

(defclass rmsprop-optimizer (memoizing-optimizer)
  ((coeff   :type single-float
            :accessor momentum-coeff
            :initarg :coeff
            :initform *momentum-coeff*))
  (:default-initargs
   :initial-value 1f-8)
  (:documentation "RMSprop optimizer"))

(defun make-optimizer (type network)
  (if (eq type 'sgd-optimizer)
      (make-instance type)
      (make-instance type :neural-network network)))

(defgeneric learn (optimizer neural-network samples))

(defmethod initialize-instance :after ((optimizer memoizing-optimizer)
                                       &rest initargs
                                       &key &allow-other-keys)
  (let ((network (getf initargs :neural-network)))
    (if (not network)
        (error "Specify a network to train"))
    (setf (optimizer-weights optimizer)
          (loop for m in (neural-network-weights network)
                collect (magicl:const
                         (optimizer-initial-value optimizer)
                         (magicl:shape m)
                         :type 'single-float))
          (optimizer-biases optimizer)
          (loop for m in (neural-network-biases network)
                collect (magicl:const
                         (optimizer-initial-value optimizer)
                         (magicl:shape m)
                         :type 'single-float)))))
