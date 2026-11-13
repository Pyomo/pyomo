Quickstart
==========

Below is a simple example written using EDI that should help get new users off the ground.  For simple problems this example case is fine to build on, but for more advanced techniques (ex: using multi-dimensional variables or advanced black-box usage) see the additional documentation.

.. note::

    The files and examples that appear in this documentation have been adapted to conform to PEP-8 standards, however, we **strongly discourage** the adoption of PEP-8 in the course of regular EDI usage as it makes the code (particularly variable declarations, constant declarations, and constraint list declarations) extremely challenging to read and debug

The example shown here minimizes a linear objective function subject to the interior area of the unit circle:

.. math::
    \begin{align*}
        & \underset{x,y,z}{\text{minimize}}
        & & x+y \\
        & \text{subject to}
        & & z = x^2 + y^2\\
        &&& z \leq 1.0 \text{ [m$^2$]}
    \end{align*}

.. literalinclude:: ../../../../pyomo/contrib/edi/examples/readme_example.py
    :language: python 
