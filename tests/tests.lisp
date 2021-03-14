;; TODO: Test different activation functions
;; TODO: Test deep networks
;; TODO: Test RMSprop

(in-package :neural-classifier-tests)
(def-suite neural-classifier-mnist :description "Test recognition of digits")

(defun run-tests ()
  ;; Set and print hyperparameters
  (setq neural-classifier:*learn-rate* 0.01
        neural-classifier:*decay-rate* (/ 10.0 50000)
        neural-classifier:*minibatch-size* 20)

  (format t "*learn-rate*=~f *decay-rate*=~f~%*momentum-coeff*=~f *minibatch-size=~d~%"
          neural-classifier:*learn-rate*
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
  (flet ((test-optimizer (optimizer)
           ;; Create a neural network with one hidden layer with 25 neurons.
           ;; Train it for 2 epochs
           (let* ((net (neural-classifier-mnist:make-mnist-classifier 25))
                  (final-results (second
                                  (let ((*standard-output* (make-broadcast-stream)))
                                    (neural-classifier-mnist:train-epochs net 2 optimizer)))))
             ;; Check that final recognition rates for train and test
             ;; sets are > a, where a is some number > 0.1
             (is (> (car final-results) 0.6))
             (is (> (cdr final-results) 0.6)))))
    (map nil #'test-optimizer '(neural-classifier:sgd-optimizer
                                neural-classifier:momentum-optimizer
                                neural-classifier:nesterov-optimizer
                                neural-classifier:adagrad-optimizer
                                #+nil
                                neural-classifier:rmsprop-optimizer))))
