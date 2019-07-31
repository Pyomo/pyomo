Installation
======================================

PyNumero is distributed within the python optimization module pyomo.

To build PyNumero from github::
	
	git clone https://github.com/Pyomo/pyomo.git
	cd pyomo/pyomo
	python setup.py install
	pip install scipy

Alternatively one can use conda::

        conda install -c conda-forge pyomo.extras

Or pip::

        pip install pyomo
	pip install scipy
	
PyNumero relies on c++ libraries for fast evaluation of NLP functions.
There are two alternatives to get/build the libraries. The first alternative,
relies on the conda-forge channel to compile and install the libraries (PREFERRED)::

        conda install -c conda-forge pynumero_libraries
	conda install -c conda-forge pymumps

The second alternative compiles the libraries with the following instructions::

        cd pyomo/contrib/pynumero/cmake/third_party/ASL
        ./get.ASL.sh
        cd solvers
        ./configurehere

if compiling from linux::

        find . -name "makefile" -exec sed -i "s/CFLAGS = -DNo_dtoa -fPIC -O/CFLAGS = -fPIC -O/g" {} \;
	make
	cd ../../../
	mkdir build
	cd build
	cmake .. 
	make
	cp asl_interface/libpynumero_* ../../extensions/lib/Linux/
	
if compiling from mac::

        find . -name "makefile" -exec sed -ie 's/CFLAGS = -DNo_dtoa -fPIC -O/CFLAGS = -fPIC -O/g' {} \;
	make
	cd ../../../
	mkdir build
	cd build
	cmake .. 
	make
	cp asl_interface/libpynumero_* ../../extensions/lib/Darwin/


  
Requirements
-------------

Python
^^^^^^^
Information on installing and using python can be found at 
https://www.python.org/.  Python distributions can also be used to manage 
the Python interface.  Python distributions include Python(x,y) (for Windows) 
and Anaconda (for Windows and Linux).

Python(x,y) can be downloaded from http://python-xy.github.io/.  

Anaconda can be downloaded from https://store.continuum.io/cshop/anaconda/.


Python packages
^^^^^^^^^^^^^^^^^
The following python packages are required for PyNumero:

* Numpy: used to support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org/
* Scipy: used to support efficient routines for numerical integration, 
  http://www.scipy.org/

Packages can be installed using pip.

