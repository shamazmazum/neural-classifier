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
@begin(list)
@item(@c(layout) is a list of positive integers which describes the
      amount of neurons in each layer (starting from input layer).)
@item(@c(activation-funcs) is a list all the elements of which are
      either @c(:sigmoid), @c(:tanh), @c(:abs), @c(:relu),
      @c(:softmax) or @c(:identity). The length of this list must be
      equal to the length of @c(layout) minus one because the input
      layer does not have an activation function. The last element
      cannot be @c(:abs) or @c(:relu) and @c(:softmax) or
      @c(:identity) can only be the last element.)
@item(@c(input-trans) is a function which is applied to an object
      passed to @c(calculate) to transform it into an input column
      (that is a matrix with the type @c(magicl:matrix/single-float)
      and the shape @c(Nx1), where @c(N) is the first number in the
      @c(layout)). For example, if we are recognizing digits from the
      MNIST set, this function can take a number of an image in the
      set and return @c(784x1) matrix.)
@item(@c(output-trans) is a function which is applied to the output of
      @c(calculate) function (that is a matrix with the type
      @c(magicl:matrix/single-float) and the shape Mx1, where M is the
      last number in the @c(layout)) to return some object with
      user-defined meaning (called a label). Again, if we are
      recognizing digits, this function transforms @c(10x1) matrix to
      a number from 0 to 9.)
@item(@c(input-trans%) is just like @c(input-trans), but is used while
      training. It can include additional transformations to extend
      your training set (e.g. it can add some noise to input data,
      rotate an input picture by a small random angle, etc.).)
@item(@c(label-trans) is a function which is applied to a label to get
      a column (that is a matrix with the type
      @c(magicl:matrix/single-float) and the shape @c(Mx1), where
      @c(M) is the last number in the @c(layout)) which is the optimal
      output from the network for this object. With digits
      recognition, this function may take a digit @c(n) and return
      @c(10x1) matrix of all zeros with exception for @c(n)-th element
      which would be @c(1f0).)
@end(list)
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
    (flet ((make-weight-matrix (rows columns)
             (magicl:rand (list rows columns)
                          :distribution (nrandom-generator
                                         :sigma (/
                                                 (sqrt
                                                  (the (single-float 0f0)
                                                       (float columns 0f0)))))
                          :type 'single-float))
           (make-bias-vector (rows)
             (magicl:rand (list rows 1)
                          :distribution (nrandom-generator)
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
             (find :identity (butlast activation-funcs))
             (find :softmax  (butlast activation-funcs)))
         (error "Incorrect activation functions"))))))

;; Normal work
(defun calculate (neural-network object)
  "Calculate output from the network @c(neural-network) for the object
@c(object). The input transformation function (specified by
@c(:input-trans) when creating a network) is applied to the @c(object)
and the output transformation function (specified by
@c(:output-trans)) is applied to output Nx1 matrix from the network."
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
           (mapc #'magicl:.+
                 matrices1
                 matrices2
                 matrices2))
         (final-weights (x grad-x)
           (magicl:.+ (magicl:scale grad-x (/ (float *minibatch-size*)))
                      (magicl:scale x *decay-rate*)))
         (final-biases (grad-x)
           (magicl:scale grad-x (/ (float *minibatch-size*)))))

    (multiple-value-bind (weights biases)
        (calculate-gradient neural-network (car samples))
      (loop
        for sample in (cdr samples) do
          (multiple-value-bind (delta-weight delta-bias)
              (calculate-gradient neural-network sample)
            (sum-matrices delta-weight weights)
            (sum-matrices delta-bias   biases)))

      (values
       (mapcar #'final-weights
               (neural-network-weights neural-network)
               weights)
       (mapcar #'final-biases biases)))))

(defmethod learn ((optimizer sgd-optimizer) neural-network samples)
  (declare (ignore optimizer))
  (flet ((update (x delta-x)
           (magicl:.- x (magicl:scale delta-x *learn-rate*) x)))
    (multiple-value-bind (delta-weight delta-bias)
        (calculate-gradient-minibatch neural-network samples)
      (let ((weights (neural-network-weights neural-network))
            (biases  (neural-network-biases  neural-network)))
        (mapc #'update weights delta-weight)
        (mapc #'update biases  delta-bias))))
  (values))

(defmethod learn ((optimizer momentum-optimizer) neural-network samples)
  (flet ((update (x delta-x accumulated-x)
           (magicl:.+ (magicl:scale delta-x *learn-rate*)
                      (magicl:scale accumulated-x (momentum-coeff optimizer))
                      accumulated-x)
           (magicl:.- x accumulated-x x)))
    (multiple-value-bind (delta-weight delta-bias)
        (calculate-gradient-minibatch neural-network samples)
      (let ((weights (neural-network-weights neural-network))
            (biases  (neural-network-biases  neural-network))
            (acc-weights (optimizer-weights optimizer))
            (acc-biases  (optimizer-biases  optimizer)))
        (mapc #'update weights delta-weight acc-weights)
        (mapc #'update biases  delta-bias   acc-biases))))
  (values))

(defmethod learn ((optimizer nesterov-optimizer) neural-network samples)
  (flet ((step1 (x accumulated-x)
           ;; Change weights & biases by accumulated value.
           ;; (Predict values for weights & biases).
           (magicl:.- x (magicl:scale accumulated-x (momentum-coeff optimizer)) x))
         (step2 (x delta-x accumulated-x)
             ;; Update accumulated value.
             ;; Set corrected values for weights and biases
             (magicl:.+ (magicl:scale accumulated-x (momentum-coeff optimizer))
                        (magicl:scale delta-x *learn-rate*)
                        accumulated-x)
             (magicl:.- x accumulated-x x)))
    (let* ((weights (neural-network-weights neural-network))
           (biases  (neural-network-biases neural-network))
           (weights-copy (mapcar #'magicl::deep-copy-tensor weights))
           (biases-copy  (mapcar #'magicl::deep-copy-tensor biases))
           (acc-weights (optimizer-weights optimizer))
           (acc-biases  (optimizer-biases  optimizer)))
      ;; Predict weights & biases
      (mapc #'step1 weights acc-weights)
      (mapc #'step1 biases  acc-biases)
      ;; Calculate gradient at predicted point
      (multiple-value-bind (delta-weight delta-bias)
          (calculate-gradient-minibatch neural-network samples)
        ;; Calculate the final position
        (mapc #'step2 weights-copy delta-weight acc-weights)
        (mapc #'step2 biases-copy  delta-bias   acc-biases))

      ;; Set the new weights & biases
      (setf (neural-network-weights neural-network) weights-copy
            (neural-network-biases  neural-network) biases-copy)))
  (values))

(defmethod learn ((optimizer adagrad-optimizer) neural-network samples)
  (flet ((update (x delta-x accumulated-x)
           (magicl:.+
            (magicl:.* delta-x delta-x)
            accumulated-x
            accumulated-x)
           (magicl:.-
            x
            (magicl:./
             (magicl:scale delta-x *learn-rate*)
             (magicl:map #'sqrt accumulated-x))
            x)))
    (multiple-value-bind (delta-weight delta-bias)
        (calculate-gradient-minibatch neural-network samples)
      (let ((weights (neural-network-weights neural-network))
            (biases  (neural-network-biases  neural-network))
            (acc-weights (optimizer-weights optimizer))
            (acc-biases  (optimizer-biases  optimizer)))
        (mapc #'update weights delta-weight acc-weights)
        (mapc #'update biases  delta-bias   acc-biases))))
  (values))

(defmethod learn ((optimizer rmsprop-optimizer) neural-network samples)
  (flet ((update (x delta-x accumulated-x)
           (let ((coeff (momentum-coeff optimizer)))
             (declare (type single-float coeff))
             (magicl:.+
              (magicl:scale (magicl:.* delta-x delta-x) (- 1.0 coeff))
              (magicl:scale accumulated-x coeff)
              accumulated-x))
           (magicl:.-
            x
            (magicl:./
             (magicl:scale delta-x *learn-rate*)
             (magicl:map #'sqrt accumulated-x))
            x)))
    (multiple-value-bind (delta-weight delta-bias)
        (calculate-gradient-minibatch neural-network samples)
      (let ((weights (neural-network-weights neural-network))
            (biases  (neural-network-biases  neural-network))
            (acc-weights (optimizer-weights optimizer))
            (acc-biases  (optimizer-biases  optimizer)))
        (mapc #'update weights delta-weight acc-weights)
        (mapc #'update biases  delta-bias   acc-biases))))
  (values))

(defun train-epoch (neural-network generator
                    &key
                      (optimizer (make-optimizer 'sgd-optimizer neural-network))
                      (learn-rate *learn-rate*)
                      (decay-rate *decay-rate*)
                      (minibatch-size *minibatch-size*))
  "Perform training of @c(neural-network) on every object returned
by the generator @c(generator). Each item returned by @c(generator)
must be in the form @c((data-object . label)) cons
pair. @c(input-trans%) and @c(label-trans) functions passes to
@c(make-neural-network) are applied to @c(car) and @c(cdr) of each
pair respectively."
  (declare (type single-float learn-rate decay-rate)
           (type neural-network neural-network)
           (type positive-fixnum minibatch-size)
           (type snakes:basic-generator generator)
           (type optimizer optimizer))
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
         (learn optimizer neural-network minibatch-samples)
         (when (zerop (rem i 1000))
           (format *standard-output* "~d... " i)
           (force-output))
       finally (terpri))))

(defun rate (neural-network generator &key (test #'eql))
  "Calculate accuracy of the @c(neural-network) (ratio of correctly
guessed samples to all samples) using testing data from the generator
@c(generator). Each item returned by @c(generator) must be a cons pair
in the form @c((data-object . label)), as with @c(train-epoch)
function. @c(test) is a function used to compare the expected label
with the label returned by the network."
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
