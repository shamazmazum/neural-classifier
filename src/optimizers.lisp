(in-package :neural-classifier)

(defclass sgd-optimizer () ()
  (:documentation "The simplest SGD optimizer"))

(defclass momentum-optimizer ()
  ((weights :type list
            :accessor optimizer-weights)
   (biases  :type list
            :accessor optimizer-biases)
   (coeff   :type single-float
            :accessor momentum-coeff
            :initarg :coeff
            :initform *momentum-coeff*))
  (:documentation "SGD optimizer with momentum"))

(defclass nesterov-optimizer (momentum-optimizer)
  ()
  (:documentation "Nesterov accelerated SGD, improvement of SGD with
momentum"))

(defun make-sgd-optimizer ()
  (make-instance 'sgd-optimizer))

(defun make-momentum-optimizer (neural-network)
  (make-instance 'momentum-optimizer
                 :neural-network neural-network))

(defun make-nesterov-optimizer (neural-network)
  (make-instance 'nesterov-optimizer
                 :neural-network neural-network))

(defgeneric learn (optimizer neural-network samples))

(defmethod initialize-instance :after ((optimizer momentum-optimizer)
                                       &rest initargs
                                       &key &allow-other-keys)
  (let ((network (getf initargs :neural-network)))
    (if (not network)
        (error "Specify a network to train"))
    (setf (optimizer-weights optimizer)
          (loop for m in (neural-network-weights network)
                collect (magicl:zeros (magicl:shape m)
                                      :type 'single-float))
          (optimizer-biases optimizer)
          (loop for m in (neural-network-biases network)
                collect (magicl:zeros (magicl:shape m)
                                      :type 'single-float)))))
