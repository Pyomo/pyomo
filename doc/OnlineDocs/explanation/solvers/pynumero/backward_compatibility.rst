Backward Compatibility
======================

While PyNumero is a third-party contribution to Pyomo, we intend to maintain
the stability of its core functionality. The core functionality of PyNumero
consists of:

1. The ``NLP`` API and ``PyomoNLP`` implementation of this API
2. HSL and MUMPS linear solver interfaces
3. ``BlockVector`` and ``BlockMatrix`` classes
4. CyIpopt and SciPy solver interfaces

Other parts of PyNumero, such as ``ExternalGreyBoxBlock`` and
``ImplicitFunctionSolver``, are experimental and subject to change without notice.
