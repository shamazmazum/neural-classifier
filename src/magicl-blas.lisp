;; These methods optimize matrix operation functions from magicl using BLAS.

(export '(sasum))

(define-compiler-macro .+ (&whole form source1 source2 &optional target)
  (declare (ignore source2))
  (when (eq source1 target)
    (warn "Inefficient use of .+: ~a" form))
  form)

(define-compiler-macro .- (&whole form source1 source2 &optional target)
  (declare (ignore source1))
  (when (eq source2 target)
    (warn "Inefficient use of .+: ~a" form))
  form)
