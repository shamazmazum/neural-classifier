(in-package :neural-classifier)
(declaim (optimize (speed 3)
                   (debug 0)
                   (compilation-speed 0)))

(defun make-neural-network (layout &key
                                     input-trans
                                     output-trans
                                     input-trans%
                                     label-trans
                                     activation-funcs)
  "Create a new neural network.


@c(layout) is a list of positive integers which describes the amount
of neurons in each layer (starting from input layer).


@c(activation-funcs) is a list all the elements of which are either
@c(:sigmoid), @c(:tanh), @c(:abs), @c(:relu) or @c(:softmax). The
length of this list must be equal to the length of @c(layout) minus
one. The last element cannot be @c(:abs) or @c(:relu). @c(:softmax)
can be only the last element.


@c(input-trans) is a function which is applied to an object passed to
@c(calculate) to transform it into an input column (that is a matrix
with the type @c(magicl:matrix/single-float) and the shape @c(Nx1),
where @c(N) is the first number in the @c(layout)). For example, if we
are recognizing digits from the MNIST set, this function can take a
number of an image in the set and return @c(784x1) matrix.


@c(output-trans) is a function which is applied to the output of
@c(calculate) function (that is a matrix with the type
@c(magicl:matrix/single-float) and the shape Mx1, where M is the last
number in the @c(layout)) to return some object with user-defined
meaning (called a label). Again, if we are recognizing digits, this
function transforms @c(10x1) matrix to a number from 0 to 9.


@c(input-trans%) is just like @c(input-trans), but is used while
training. It can include additional transformations to extend your
training set (e.g. it can add some noise to resulting vector, rotate a
picture by a small random angle, etc.).


@c(label-trans) is a function which is applied to a label to get a
column (that is a matrix with the type @c(magicl:matrix/single-float)
and the shape @c(Mx1), where @c(M) is the last number in the
@c(layout)) which is optimal output from the network for this
object. With digits recognition, this function may take a digit @c(n)
and return @c(10x1) matrix of all zeros with exception for @c(n)-th
element which would be @c(1f0).


Default value for all transformation functions is @c(identity)."
  (let ((arguments
         `(,@(if input-trans      `(:input-trans      ,input-trans))
           ,@(if output-trans     `(:output-trans     ,output-trans))
           ,@(if input-trans%     `(:input-trans%     ,input-trans%))
           ,@(if label-trans      `(:label-trans      ,label-trans))
           ,@(if activation-funcs `(:activation-funcs ,activation-funcs)))))
    (apply #'make-instance 'neural-network :layout layout arguments)))

(defmethod initialize-instance :after ((neural-network neural-network) &rest initargs)
  (declare (ignore initargs))
  (let ((layout (neural-network-layout neural-network))
        (activation-funcs (neural-network-activation-funcs neural-network)))
    (declare (type list layout activation-funcs))
    (flet ((make-weight-matrix (rows columns)
             (magicl:rand (list rows columns)
                          :distribution (lambda ()
                                          (random-normal
                                           :sigma
                                           (/ (sqrt (the (single-float 0f0)
                                                         (float columns 0f0))))))
                          :type 'single-float))
           (make-bias-vector (rows)
             (magicl:rand (list rows 1)
                          :distribution #'random-normal
                          :type 'single-float)))
      (setf (neural-network-weights neural-network)
            (mapcar #'make-weight-matrix
                    (cdr layout) layout)
            (neural-network-biases neural-network)
            (mapcar #'make-bias-vector
                    (cdr layout))))

    (let ((n (1- (length layout))))
      (cond
        ((null activation-funcs)
         (setf (neural-network-activation-funcs neural-network)
               (loop repeat n collect :tanh)))
        ((or (/= (length activation-funcs) n)
             (eq (car (last activation-funcs)) :abs)
             (eq (car (last activation-funcs)) :relu)
             (find :softmax (butlast activation-funcs)))
         (error "Incorrect activation functions"))))))

;; Normal work
(defun calculate (neural-network object)
  "Calculate the output from the network @c(neural-network) for the
object @c(object). The input transformation function is applied to the
@c(object) and the output transformation function is applied to the
output column from the network."
  (declare (type neural-network neural-network))
  (let ((weights (neural-network-weights neural-network))
        (biases  (neural-network-biases  neural-network))
        (activation-funcs (neural-network-activation-funcs neural-network))
        (input-trans (neural-network-input-trans neural-network))
        (output-trans (neural-network-output-trans neural-network)))
    (declare (type function input-trans output-trans))
    (flet ((calculate-layer (input layer)
             (destructuring-bind (weights biases activation)
                 layer
               (activation
                (magicl:.+ (magicl:@ weights input)
                           biases)
                activation))))
      (funcall
       output-trans
       (reduce #'calculate-layer
               (mapcar #'list weights biases activation-funcs)
               :initial-value (funcall input-trans object))))))

;; Training
(defun calculate-z-and-out (neural-network input)
  "Calculate argument and value of activation function for all layers"
  (declare (type neural-network neural-network)
           (type magicl:matrix/single-float input))
  (labels ((accumulate-z-and-out (layers input z-acc out-acc)
             (if layers
                 (destructuring-bind (weights biases activation)
                     (car layers)
                   (let* ((z (magicl:.+ (magicl:@ weights input) biases))
                          (out (activation z activation)))
                     (accumulate-z-and-out
                      (cdr layers)
                      out
                      (cons z z-acc)
                      (cons out out-acc))))
                 (values
                  z-acc
                  out-acc))))
    ;; Output is in backward order: input layer last
    (accumulate-z-and-out
     (mapcar #'list
             (neural-network-weights neural-network)
             (neural-network-biases neural-network)
             (neural-network-activation-funcs neural-network))
     input
     nil nil)))

(defun calculate-delta (neural-network z network-output expected-output)
  "Calculate partial derivative of the cost function by z for all layers"
  (declare (type magicl:matrix/single-float expected-output))
  (labels ((backprop (layer acc)
             (if layer
                 (destructuring-bind (w-l+1 activation-l z-l)
                     (car layer)
                   (backprop (cdr layer)
                             (cons
                              (magicl:.* (activation-derivative z-l activation-l)
                                         (magicl:mult w-l+1 (car acc) :transa :t))
                              acc)))
                 acc)))
    (backprop
     (mapcar #'list
             (reverse (neural-network-weights neural-network))
             (cdr (reverse (neural-network-activation-funcs neural-network)))
             (cdr z))
     (list
      (magicl:.- network-output
                 expected-output)))))

(defun calculate-gradient (neural-network sample)
  "Calculate gradient of the cost function"
  (declare (type cons sample))
  (let ((input    (funcall
                   (the function (neural-network-input-trans% neural-network))
                   (car sample)))
        (expected (funcall
                   (the function (neural-network-label-trans neural-network))
                   (cdr sample))))
    (declare (type magicl:matrix/single-float input expected))
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
                   matrices1
                   matrices2
                   matrices2)))
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
             (declare (type single-float decay))
             (lambda (x delta-x)
               (declare (type magicl:matrix/single-float x delta-x))
               (magicl:.-
                (magicl:.* (- 1f0 (* *learn-rate* decay)) x)
                (magicl:.* (/ *learn-rate* *minibatch-size*) delta-x)))))
      (with-accessors ((weights neural-network-weights)
                       (biases  neural-network-biases))
          neural-network
        (setf weights (mapcar (improver *decay-rate*) weights delta-weight)
              biases  (mapcar (improver 0f0)          biases  delta-bias))))))

(defun train-epoch (neural-network generator
                    &key
                      (learn-rate *learn-rate*)
                      (decay-rate *decay-rate*)
                      (minibatch-size *minibatch-size*))
  "Perform a training of @c(neural-network) on every object returned
by the generator @c(generator). Each item returned by @c(generator)
must be a cons pair containing an object which is passed to the neural
network and its label. @c(input-trans%) and @c(label-trans) functions
passed to @c(make-neural-network) are applied to @c(car) and @c(cdr)
of each cons pair."
  (declare (type single-float learn-rate decay-rate)
           (type neural-network neural-network)
           (type positive-fixnum minibatch-size)
           (type snakes:basic-generator generator))
  (let ((*learn-rate* learn-rate)
        (*decay-rate* decay-rate)
        (*minibatch-size* minibatch-size))
    (loop
       for minibatch-samples =
         (snakes:take *minibatch-size* generator
                      :fail-if-short nil)
       for i fixnum from 0 by *minibatch-size*
       while minibatch-samples
       do
         (learn neural-network minibatch-samples)
         (when (zerop (rem i 1000))
           (format *standard-output* "~d... " i)
           (force-output))
       finally (terpri))))

(defun rate (neural-network generator &key (test #'eql))
  "Calculate accuracy of the @c(neural-network) (that is a ratio of
correctly guessed samples to all samples) using testing data from
the generator @c(generator). Each item returned by @c(generator) must
be a cons pair containing an object which is passed to the network and
its label. @c(test) is a function used to compare the expected label
and the actual one."
  (declare (type snakes:basic-generator generator)
           (type function test))
  (labels ((calculate-accuracy (hits total)
             (declare (type fixnum hits total))
             (let ((sample (funcall generator)))
               (if (not (eq sample 'snakes:generator-stop))
                   (calculate-accuracy
                    (+ (if (funcall test
                                    (calculate neural-network (car sample))
                                    (cdr sample))
                           1 0)
                       hits)
                    (1+ total))
                   (float (/ hits total))))))
    (calculate-accuracy 0 0)))

#+sbcl
(progn
  (format t "Disabling floating point traps~%")
  (sb-int:set-floating-point-modes :traps '(:divide-by-zero)))
#-sbcl
(progn
  (format t "You may wish to disable floating point traps, especially overflow~%"))
