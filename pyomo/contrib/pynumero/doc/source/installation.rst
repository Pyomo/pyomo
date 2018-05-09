Installation
======================================

PyNumero can be installed as a python package using pip or from source using git.  
More information on pip can be found at https://pypi.python.org/pypi/pip.
More information on git can be found at http://git-scm.com. 

To install using pip (**NOT COMPLETE**)::

	pip install pynumero
	
To build PyNumero from github

	git clone https://TODO
	cd pynumero
	pip install -r requirements.txt
	python setup.py install

Developers can build PyNumero in development mode::
	
	git clone https://TODO
	cd pynumero
	pip install -r requirements.txt
	python setup.py develop
	
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
The following python packages are required for WNTR:

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

The following python packages are optional for WNTR:

* Numpydoc [vanderWalt2011]_: used to build the user manual,
  https://github.com/numpy/numpydoc

* HSL [HSL2013]_: solvers for Ipopt, http://www.hsl.rl.ac.uk/ipopt/.
	
  * Select Windows or Linux in the COIN-HSL Archive, Personal License box
  * Select Personal License, fill out the form and accept
  * Download the zip file from the link sent via email
  TODO: say something here about compiling hsl with pynumero
