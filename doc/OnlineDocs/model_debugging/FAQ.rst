FAQ
===

#. Solver not found

Solvers are **not** distributed with Pyomo and must be installed
separately by the user. In general, the solver executable must be accessible using a terminal command. For example, ipopt can only be used as a solver if
the command

::

   $ ipopt

invokes the solver. For example

::

   $ ipopt -?
   usage: ipopt [options] stub [-AMPL] [<assignment> ...]

   Options:
   	--  {end of options}
	-=  {show name= possibilities}
	-?  {show usage}
	-bf {read boundsfile f}
	-e  {suppress echoing of assignments}
	-of {write .sol file to file f}
	-s  {write .sol file (without -AMPL)}
	-v  {just show version}
