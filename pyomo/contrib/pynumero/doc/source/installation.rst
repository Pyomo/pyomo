Installation
======================================

PyNumero is distributed within the python optimization module pyomo.

To build PyNumero from github (currently on a fork)::
	
	git clone https://github.com/santiagoropb/pyomo.git
	cd pyomo/pyomo
	git checkout pynumero_release2
	python setup.py develop

PyNumero relies on c++ libraries for fast evaluation of NLP functions.
Future releases of PyNumero will distribute binaries via conda-forge and pip.
However, the current release requires the user to compile the libraries.

There are two alternatives to compile the libraries. The first alternative,
relies on linking the PyNumero libraries against ASL from ampl-mp. If conda
is available, ampl-mp can be installed with the following command::

        conda install -c conda-forge ampl-mp

Then one can compile the libraries as follows::

        cd pyomo/contrib/pynumero/cmake
        mkdir build
        cd build
        mp_dir=$(find ~/ -type d -name "ampl-mp*")
        cmake .. -DMP_PATH=$mp_dir
        make
	cp lib/libpynumero_* ../../extensions/lib/Linux/

The second alternative compiles ASL and links it against the pynumero libraries::

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
	cp lib/libpynumero_* ../../extensions/lib/Linux/
	
if compiling from mac::

        find . -name "makefile" -exec sed -ie 's/CFLAGS = -DNo_dtoa -fPIC -O/CFLAGS = -fPIC -O/g' {} \;
	make
	cd ../../../
	mkdir build
	cd build
	cmake .. 
	make
	cp lib/libpynumero_* ../../extensions/lib/Darwin/


  
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

Python distributions include several tools for code development (i.e. Spyder, SciTE), 
numerical computations, data analysis and visualization. 
Spyder is an interactive development environment that includes enhanced 
editing and debug features along with a layout that is very similar 
to using MATLAB. Debugging features are also available from the toolbar.  
Code documentation is displayed in the object inspection 
window, pop-up information on class structure and functions is displayed in the 
editor and console windows.  
SciTE is a cross platform text editor designed for 
editing code.  SciTE recognizes many languages (including Python and YML) and 
includes syntax highlighting, indentation, and function recognition. 

Python packages
^^^^^^^^^^^^^^^^^
The following python packages are required for PyNumero:

* Numpy [vanderWalt2011]_: used to support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org/
* Scipy [vanderWalt2011]_: used to support efficient routines for numerical integration, 
  http://www.scipy.org/
* NetworkX [Hagberg2008]_: used to create and analyze complex networks, 
  https://networkx.github.io/
* Pandas [McKinney2013]_: used to analyze and store time series data, 
  http://pandas.pydata.org/
* Matplotlib [Hunter2007]_: used to produce figures, 
  http://matplotlib.org/

Packages can be installed using pip.

Optional dependencies
-------------------------

PyNumero requires the following software for compilation of libraries:

* cmake: used to build libraries. If anaconda is available, cmake can be installed from
  conda-forge

  conda install -c conda-forge cmake

* Compilers: C, C++ and gfortran compilers

* ampl-mp (optional): used in the compilation of binaries. If conda is available, ampl-mp
  can be installed from conda-forge

  conda install -c conda-forge ampl-mp 
