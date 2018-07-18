GETTING STARTED
---------------

0.  Install Sphinx

  pip install sphinx

1. Edit documentation

  vi *.rst

2.  Build the documentation

  make html

or

  make latexpdf

NOTE:  If the local python is not on your path, then you may need to 
invoke 'make' differently.  For example, using the PyUtilib 'lbin' command:

  lbin make html

3.  Admire your work

  cd _build/html
  open index.html

4.  Repeat

  GOTO STEP 1
