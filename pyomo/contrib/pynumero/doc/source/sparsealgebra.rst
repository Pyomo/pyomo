Sparse Linear Algebra and Matrix Vector Storage
===============================================

Matrix vector operations are fundamental for the development of any
numerical algorithm. Numpy is a popular Python package that provides
functionality to store and manipulate n-dimensional arrays of data, with
most of the operations being performed in compiled code. Over the years,
the efficiency and flexibility of Numpy has made the package an
essential library for most of today's scientific/mathematical
Python-based software. Scipy is another popular Python package for
scientific computing which builds on Numpy to provide a collection of
common numerical routines (also pre-compiled) and sparse matrix storage
schemes. To exploit the capabilities of the Numpy/ Scipy ecosystem,
PyNumero stores vectors and matrices from the NLP interface in Numpy
arrays and Scipy sparse matrices. By doing so, PyNumero benefits from
the fast pre-compiled operations of Numpy (e.g. vectorization and
broadcasting), makes all subroutines in Numpy/Scipy available for
implementing optimization algorithms, and minimizes the burden on users
to learn additional syntax besides what is offered in Numpy/Scipy.

See the `Numpy documentation <https://numpy.org/>`_ and the `Scipy
documentation <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
to learn more about the vast capabilities of Numpy and Scipy. In
particular, we recommend reading about the ``numpy.ndarray`` class,
``scipy.sparse`` matrices, and the ``numpy.linalg`` and ``scipy.sparse``
modules.
