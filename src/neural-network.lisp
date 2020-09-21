(in-package :neural-classifier)
(declaim (optimize (speed 3)
                   (debug 0)
                   (compilation-speed 0)))

(deftype non-negative-fixnum () '(integer 0 #.most-positive-fixnum))

(defmacro make-neural-network (layout &key input-trans output-trans train-trans)
  "Create a neural network. LAYOUT describes the amount of neurons in each layer."
  `(make-instance 'neural-network
                  :layout ,layout
                  ,@(if input-trans  `(:input-trans  ,input-trans))
                  ,@(if output-trans `(:output-trans ,output-trans))
                  ,@(if train-trans  `(:train-trans  ,train-trans))))

(defmethod initialize-instance :after ((neural-network neural-network) &rest initargs)
  (declare (ignore initargs))
  (let ((layout (neural-network-layout neural-network)))
    (flet ((make-weight-matrix (rows columns)
             (magicl:rand (list rows columns)
                          :distribution (nrandom-generator
                                         :sigma (/
                                                 (sqrt
                                                  (the (double-float 0d0)
                                                       (float columns 0d0)))))))
           (make-bias-vector (rows)
             (magicl:rand (list rows 1)
                          :distribution (nrandom-generator))))
      (setf (neural-network-weights neural-network)
            (mapcar #'make-weight-matrix
                    (cdr layout) layout)
            (neural-network-biases neural-network)
            (mapcar #'make-bias-vector
                    (cdr layout))))))

;; Normal work
(defun calculate (neural-network object)
  "Calculate the output from the network NEURAL-NETWORK for the object OBJECT."
  (declare (type neural-network neural-network))
  (let ((weights (neural-network-weights neural-network))
        (biases  (neural-network-biases  neural-network))
        (input-trans (neural-network-input-trans neural-network))
        (output-trans (neural-network-output-trans neural-network)))
    (declare (type function input-trans output-trans))
    (flet ((calculate-layer (input weights-and-biases)
             (destructuring-bind (weights . biases)
                 weights-and-biases
               (magicl:map #'sigma
                           (magicl:.+ (magicl:@ weights input)
                                      biases)))))
      (funcall
       output-trans
       (reduce #'calculate-layer
               (mapcar #'cons weights biases)
               :initial-value (funcall input-trans object))))))

;; Training
(defun calculate-z-and-out (neural-network input)
  "Calculate argument and value of sigma for all layers"
  (declare (type neural-network neural-network)
           (type magicl:matrix/double-float input))
  (labels ((accumulate-z-and-out (weights biases input z-acc out-acc)
             (if (and weights biases)
                 (let* ((weight (car weights))
                        (weight-rest (cdr weights))
                        (bias (car biases))
                        (bias-rest (cdr biases))
                        (z (magicl:.+ (magicl:@ weight input) bias))
                        (out (magicl:map #'sigma z)))
                   (accumulate-z-and-out
                    weight-rest
                    bias-rest
                    out
                    (cons z z-acc)
                    (cons out z-acc)))
                 (values
                  z-acc
                  out-acc))))
    ;; Output is in backward order: input layer last
    (accumulate-z-and-out
     (neural-network-weights neural-network)
     (neural-network-biases neural-network)
     input
     nil nil)))

(defun calculate-delta (neural-network z network-output expected-output)
  "Calculate partial derivative of the cost function by z for all layers"
  (declare (type magicl:matrix/double-float expected-output))
  (labels ((backprop (weight z acc)
             (if z
                 (let ((delta-l+1 (car acc))
                       (z-l (car z))
                       (w-l+1 (car weight)))
                   (backprop (cdr weight)
                             (cdr z)
                             (cons
                              (magicl:.* (magicl:map #'sigma% z-l)
                                         (magicl:@ (magicl:transpose w-l+1) delta-l+1))
                              acc)))
                 acc)))
    (backprop
     (reverse (neural-network-weights neural-network))
     (cdr z)
     (list
      (magicl:.- network-output
                 expected-output)))))

(defun calculate-gradient (neural-network input expected-output)
  "Calculate gradient of the cost function"
  (declare (type magicl:matrix/double-float input expected-output))
  (multiple-value-bind (z output)
      (calculate-z-and-out neural-network input)
    (let ((delta (calculate-delta
                  neural-network z
                  (car output) expected-output))
          (output (cons input (reverse output))))
    (flet ((weight-grad (a delta)
             (magicl:transpose (magicl:@ a (magicl:transpose delta)))))
      (values
       (mapcar #'weight-grad output delta) ;; Weights
       delta)))))                          ;; Biases

(defun learn (neural-network provider)
  (declare (type snakes:basic-generator provider))
  (multiple-value-bind (input expected-output)
      (funcall provider)
    (when (not (eq input 'snakes:generator-stop))
      (let ((input-trans (neural-network-input-trans neural-network))
            (train-trans (neural-network-train-trans neural-network)))
        (declare (type function input-trans train-trans))
        (multiple-value-bind (delta-weight delta-bias)
            (calculate-gradient neural-network
                                (funcall input-trans input)
                                (funcall train-trans expected-output))
          (flet ((improver (decay)
                   (declare (type double-float decay))
                   (lambda (x delta-x)
                     (declare (type magicl:matrix/double-float x delta-x))
                     (magicl:.-
                      (multiply-by-scalar (- 1d0 (* *learn-rate* decay)) x)
                      (multiply-by-scalar *learn-rate* delta-x)))))
            (with-accessors ((weights neural-network-weights)
                             (biases  neural-network-biases))
                neural-network
              (setf weights (mapcar (improver *decay-rate*) weights delta-weight)
                    biases  (mapcar (improver 0d0)          biases  delta-bias))))))
      t)))

(defun train-epoch (neural-network provider
                    &optional
                      (learn-rate *learn-rate*)
                      (decay-rate *decay-rate*))
  "Train neural network on N objects provided by PROVIDER."
  (declare (type double-float learn-rate decay-rate)
           (type neural-network neural-network))
  (let ((*learn-rate* learn-rate)
        (*decay-rate* decay-rate))
    (loop
       for i fixnum from 0 by 1
       while (learn neural-network provider)
       do
         (when (zerop (rem i 1000))
           (format *standard-output* "~d... " i)
           (force-output))
       finally (terpri))))

(defun rate (neural-network provider &key (test #'eql))
  "Calculate ratio of correct guesses based on N samples from the PROVIDER."
  (declare (type snakes:basic-generator provider)
           (type function test))
  (labels ((rate% (matches total)
             (declare (type non-negative-fixnum matches total))
             (multiple-value-bind (input expected-output)
                 (funcall provider)
               (if (eq input 'snakes:generator-stop)
                   (float (/ matches total))
                   (rate%
                    (+ matches
                       (if (funcall test
                                    (calculate neural-network input)
                                    expected-output)
                           1 0))
                    (1+ total))))))
    (rate% 0 0)))
