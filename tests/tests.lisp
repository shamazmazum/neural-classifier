;; TODO: Test different activation functions
;; TODO: Test deep networks
;; TODO: Test RMSprop

(in-package :neural-classifier-tests)
(def-suite neural-classifier-mnist :description "Test recognition of digits")

(defun run-tests ()
  ;; Set and print hyperparameters
  (setq neural-classifier:*decay-rate* (/ 10.0 50000)
        neural-classifier:*minibatch-size* 20)

  (format t "*decay-rate*=~f *momentum-coeff*=~f *minibatch-size*=~d~%"
          neural-classifier:*decay-rate*
          neural-classifier:*momentum-coeff*
          neural-classifier:*minibatch-size*)

  ;; Load MNIST dataset
  (neural-classifier-mnist:load-mnist-database)

  (every
   #'identity
   (mapcar (lambda (suite)
             (explain! (run suite)))
           '(neural-classifier-mnist))))

(in-suite neural-classifier-mnist)
(test optimizers
  (flet ((test-optimizer (optimizer &rest parameters)
           ;; Create a neural network with one hidden layer with 25 neurons.
           ;; Train it for 2 epochs
           (let* ((net (neural-classifier-mnist:make-mnist-classifier 25))
                  (final-results (second
                                  (let ((*standard-output* (make-broadcast-stream)))
                                    (neural-classifier-mnist:train-epochs
                                     net 2
                                     (apply #'make-instance optimizer
                                            (concatenate
                                             'list
                                             (if (not (eq optimizer
                                                          'neural-classifier:sgd-optimizer))
                                                 (list :neural-network net))
                                             parameters)))))))
             ;; Check that final recognition rates for train and test
             ;; sets are > a, where a is some number > 0.1
             (is (> (car final-results) 0.6))
             (is (> (cdr final-results) 0.6)))))
    (test-optimizer 'neural-classifier:sgd-optimizer      :learning-rate 1f-2)
    (test-optimizer 'neural-classifier:momentum-optimizer :learning-rate 1f-2)
    (test-optimizer 'neural-classifier:nesterov-optimizer :learning-rate 1f-2)
    (test-optimizer 'neural-classifier:adagrad-optimizer  :learning-rate 1f-2)
    (test-optimizer 'neural-classifier:rmsprop-optimizer  :learning-rate 1f-3)))
