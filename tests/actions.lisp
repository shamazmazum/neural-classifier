(defun do-all()
  (ql:quickload :neural-classifier/tests)
  (uiop:quit
   (if (uiop:symbol-call
        :neural-classifier-tests
        '#:run-tests)
       0 1)))

(do-all)
