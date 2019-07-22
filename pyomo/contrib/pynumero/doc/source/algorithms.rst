Writing Nonlinear Optimization Algorithms
=========================================

Writing nonlinear optimization algorithms is challenging. In general, the implementation of these algorithms requires a good understanding of the mathematical theory as well of software development. With PyNumero we hope to motivate students in the field of nonlinear optimization to write and test their own algorithms. We have implemented different examples for users to get familiar with PyNumero. All these examples utilize the building blocks described in the previous sections. Among the algorithms implemented in PyNumero we have

* Newton for solving nonlinear system of equations
* Mehrotras algorithm for convex QPs
* Interior-Point (Ipopts algorithm)
* Penalty-Interior-Point (WORHPs algorithm from Renke Kuhlmann)

Another good tool for implementing nonlinear optimization algorithms in PyNumero is the **NLPState** interface.
