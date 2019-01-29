Pyomo Interface to MC++
=======================

The Pyomo-MC++ interface allows for bounding of factorable functions using the MC++ library developed by
the OMEGA research group at Imperial College London.
Documentation for MC++ may be found on the `MC++ website`_.

.. _MC++ website: https://dx.doi.org/10.1016/0098-1354(95)00219-7

Installation
------------

Support for MC++ has only been validated by Pyomo developers using Linux and OSX.
Installation instructions for the MC++ library may be found on the `MC++ website`_.
Until a more simple installation scheme is developed, you may install MC++ and this interface using this guide.

We assume that you have installed MC++ into a directory of your choice.
We will denote this directory by ``$MCPP_PATH``.
For example, you should see that the file ``$MCPP_PATH/INSTALL`` exists.

Navigate to the ``pyomo/contrib/mcpp`` directory in your pyomo installation.
This directory should contain a file named ``mcppInterface.cpp``.
You will need to compile this file using the following command:

.. code::

    g++ -I $MCPP_PATH/src/3rdparty/fadbad++ -I $MCPP_PATH/src/mc -I /usr/include/python3.6 -fPIC -O2 -c mcppInterface.cpp

This links the MC++ required library FADBAD++, MC++ itself, and Python to compile the Pyomo-MC++ interface.
If successful, you will now have a file named ``mcppInterface.o`` in your working directory.
If you are not using Python 3.6, you will need to link to the appropriate Python version.
You now need to create a shared object file with the following command:

.. code::

    g++ -shared mcppInterface.o -o mcppInterface.so

You may then test your installation by running the test file:

.. code::

    python test_mcpp.py
