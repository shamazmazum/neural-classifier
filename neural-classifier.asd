(defsystem :neural-classifier
  :name :neural-classifier
  :version "0.1"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Classification of samples based on neural network."
  :licence "2-clause BSD"
  :pathname "src/"
  :serial t
  :components ((:file "package")
               (:file "definitions")
               (:file "utility")
               (:file "neural-network"))
  :depends-on (:magicl))

(defsystem :neural-classifier/digits
  :name :neural-classifier/digits
  :version "0.1"
  :author "Vasily Postnicov <shamaz.mazum at gmail dot com>"
  :description "Recognition of handwritten digits based on MNIST dataset."
  :licence "2-clause BSD"
  :pathname "digits/"
  :serial t
  :components ((:file "package")
               (:file "digits"))
  :depends-on (:neural-classifier :nibbles))
