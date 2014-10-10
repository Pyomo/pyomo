#!/bin/bash

# this script is for a particular SGE machine, but the mpiexec command
# would be similar on any machine with mpi
# this assumes you are launching from the networkflow directory

# it is set up to do the 5 bundle example using 5 solver servers
# followed by solving the ef 

#$ -N runnetflow
#$ -cwd
#$ -V
#$ -o runnetflow.out
#$ -e runnetflow.err       

mpiexec \
-np 1 pyomo_ns : \
-np 1 dispatch_srvr : \
-np 5 phsolverserver : \
-np 1 runph --solver=cplex --solver-manager=phpyro --shutdown-pyro -m models -i 1ef10 --solver=cplex --rho-cfgfile=config/rhosetter0.5.cfg --bounds-cfgfile=config/xboundsetter.cfg --max-iterations=20 --scenario-solver-options="threads=4"  --scenario-bundle-specification=10scenario-bundle-specs/FiveBundles.dat --output-times --solve-ef --ef-mipgap=0.01 --output-ef-solver-log --enable-ww-extensions --ww-extension-cfgfile=config/wwph-fixlag10.cfg --ww-extension-suffixfile=config/wwph.suffixes --linearize-nonbinary-penalty-terms=4 >& netflow.out
