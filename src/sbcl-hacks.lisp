;; Stuff for optimizing math functions for single-float type.
;; Should be integrated in SBCL itself, but it's hard to do it right.
(defpackage neural-classifier-sbcl
  (:use :cl))
(in-package :neural-classifier-sbcl)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun symbolicate (&rest args)
    (intern
     (apply
      #'concatenate
      'string
      (mapcar
       (lambda (x)
         (typecase x
           (string (string-upcase x))
           (symbol (symbol-name x))
           (t (error "Not a symbol or string"))))
       args)))))

;; Alien mathematical functions (look at src/code/irrat.lisp in SBCL sources)
(macrolet
    ((def-alien (name n-args)
       (let ((func-name (symbolicate "%" name "f"))
             (alien-name (format nil "~af" name))
             (args (loop for i below n-args collect
                        (intern (format nil "ARG~d" i)))))
         `(progn
            (declaim (inline ,func-name))
            (defun ,func-name ,args
              (sb-ext:truly-the
               single-float
               (sb-alien:alien-funcall
                (sb-alien:extern-alien
                 ,alien-name
                 (function single-float
                           ,(loop repeat n-args collect 'single-float)))
                ,@args)))))))
  (def-alien "exp" 1)
  (def-alien "tanh" 1))

;; Tell the compiler these functions are pure
;; (look at src/compiler/generic/vm-fndb.lisp in SBCL source code).

;; Additional hack is needed for these functions to be really flushable:
;; https://sourceforge.net/p/sbcl/mailman/message/37134684/
(sb-c:defknown (%exp %tanh)
    (single-float) single-float
    (sb-c:movable sb-c:flushable sb-c:foldable))

;; Define IR1 transformations from EXP to %EXP and so on.
;; (look at src/compiler/float-tran.lisp in SBCL source code).
(macrolet
    ((def-trans (name n-args)
       (let ((arg-types (loop repeat n-args collect 'single-float))
             (args (loop for i below n-args collect
                        (intern (format nil "ARG~d" i))))
             (trans-name (symbolicate "%" name "f")))
         `(sb-c:deftransform ,name (,args ,arg-types *)
            '(,trans-name ,@args)))))
  (def-trans exp 1)
  (def-trans tanh 1))
