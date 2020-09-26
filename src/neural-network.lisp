(in-package :neural-classifier)
(declaim (optimize (speed 3)
                   (debug 0)
                   (compilation-speed 0)))

(defmacro make-neural-network (layout &key input-trans output-trans train-trans)
  "Create a new neural network.


@c(layout) is a list of positive integers which describes the amount
of neurons in each layer (starting from input layer).


@c(input-trans) is a function which is applied to an object passed to
@c(calculate) to transform it into an input column (that is a matrix
with the type @c(magicl:matrix/double-float) and the shape @c(Nx1),
where @c(N) is the first number in the @c(layout)). For example, if we
are recognizing digits from the MNIST set, this function can take a
number of an image in the set and return @c(784x1) matrix.


@c(output-trans) is a function which is applied to the output of
@c(calculate) function (that is a matrix with the type
@c(magicl:matrix/double-float) and the shape Mx1, where M is the last
number in the @c(layout)) to return some object with user-defined
meaning. Again, if we are recognizing digits, this function transforms
@c(10x1) matrix to a number from 0 to 9.


@c(train-trans) is a function which is applied to an object from the
train set to get a column (that is a matrix with the type
@c(magicl:matrix/double-float) and the shape @c(Mx1), where @c(M) is
the last number in the @c(layout)) which is optimal output from the
network for this object. With digits recognition, this function may
take a digit @c(n) and return @c(10x1) matrix of all zeros with
exception for @c(n)-th element which would be @c(1d0).


Default value for all transformation functions is @c(identity)."
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
  "Calculate the output from the network @c(neural-network) for the
object @c(object). The input transformation function is applied to the
@c(object) and the output transformation function is applied to the
output column from the network."
  (declare (type neural-network neural-network))
  (let ((weights (neural-network-weights neural-network))
        (biases  (neural-network-biases  neural-network))
        (input-trans (neural-network-input-trans neural-network))
        (output-trans (neural-network-output-trans neural-network)))
    (declare (type function input-trans output-trans))
    (flet ((calculate-layer (input weights-and-biases)
             (destructuring-bind (weights . biases)
                 weights-and-biases
               (magicl:map! #'sigma
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
                    (cons out out-acc)))
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
                                         (magicl:mult w-l+1 delta-l+1 :transa :t))
                              acc)))
                 acc)))
    (backprop
     (reverse (neural-network-weights neural-network))
     (cdr z)
     (list
      (magicl:.- network-output
                 expected-output)))))

(defun calculate-gradient (neural-network sample)
  "Calculate gradient of the cost function"
  (declare (type cons sample))
  (let ((input    (funcall
                   (the function (neural-network-input-trans neural-network))
                   (car sample)))
        (expected (funcall
                   (the function (neural-network-train-trans neural-network))
                   (cdr sample))))
    (declare (type magicl:matrix/double-float input expected))
    (multiple-value-bind (z output)
        (calculate-z-and-out neural-network input)
      (let ((delta (calculate-delta
                    neural-network z
                    (car output) expected))
            (output (cons input (reverse output))))
        (flet ((weight-grad (a delta)
                 (magicl:mult delta a :transb :t)))
          (values
           (mapcar #'weight-grad output delta) ;; Weights
           delta))))))                         ;; Biases

(defun calculate-gradient-minibatch (neural-network samples)
  "Calculate gradient of the cost function based on multiple input samples"
  (declare (type list samples))
  (flet ((sum-matrices (matrices1 matrices2)
           (declare (type list matrices1 matrices2))
           (mapcar #'magicl:.+
                   matrices1 matrices2)))
    (loop
       with weights = nil
       with biases = nil
       for sample in samples
       do
         (multiple-value-bind (delta-weight delta-bias)
             (calculate-gradient neural-network sample)
           (push delta-weight weights)
           (push delta-bias biases))
       finally
         (return
           (values
            (reduce #'sum-matrices weights)
            (reduce #'sum-matrices biases))))))

(defun learn (neural-network samples)
  (multiple-value-bind (delta-weight delta-bias)
      (calculate-gradient-minibatch neural-network samples)
    (flet ((improver (decay)
             (declare (type double-float decay))
             (lambda (x delta-x)
               (declare (type magicl:matrix/double-float x delta-x))
               (magicl:.-
                (magicl:.* (- 1d0 (* *learn-rate* decay)) x)
                (magicl:.* (/ *learn-rate* *minibatch-size*) delta-x)))))
      (with-accessors ((weights neural-network-weights)
                       (biases  neural-network-biases))
          neural-network
        (setf weights (mapcar (improver *decay-rate*) weights delta-weight)
              biases  (mapcar (improver 0d0)          biases  delta-bias))))))

(defun train-epoch (neural-network samples
                    &key
                      (learn-rate *learn-rate*)
                      (decay-rate *decay-rate*)
                      (minibatch-size *minibatch-size*))
  "Perform a training of @c(neural-network) on every object from the
list @c(samples). Each item in @c(samples) must be a cons pair
containing an object which is passed to the neural network and the
expected output for that object (after the output transformation)."
  (declare (type double-float learn-rate decay-rate)
           (type neural-network neural-network)
           (type positive-fixnum minibatch-size)
           (type list samples))
  (let ((*learn-rate* learn-rate)
        (*decay-rate* decay-rate)
        (*minibatch-size* minibatch-size))
    (loop
       for current-size = (min (length samples)
                               *minibatch-size*)
       for minibatch-samples = (subseq samples 0 current-size)
       for i fixnum from 0 by *minibatch-size*
       while minibatch-samples
       do
         (learn neural-network minibatch-samples)
         (setq samples (subseq samples current-size))
         (when (zerop (rem i 1000))
           (format *standard-output* "~d... " i)
           (force-output))
       finally (terpri))))

(defun rate (neural-network samples &key (test #'eql))
  "Calculate accuracy of the @c(neural-network) (that is a ratio of
correctly guessed samples to all samples) using testing data from
the list @c(samples). Each item in @c(samples) must be a cons pair
containing an object which is passed to the network and the expected
output for that object (after the output transformation). @c(test) is
a function used to compare the expected output and the actual one."
  (declare (type list samples)
           (type function test))
  (loop
     for sample in samples
     for input = (car sample)
     for expected = (cdr sample)
     count
       (funcall test
                (calculate neural-network input)
                expected)
     into positive
     finally (return (float (/ positive (length samples))))))

#+sbcl
(progn
  (format t "Disabling floating point traps~%")
  (sb-int:set-floating-point-modes :traps '(:divide-by-zero)))
#-sbcl
(progn
  (format t "You may wish to disable floating point traps, especially overflow~%"))
