#!/bin/bash
# Must be run from coopr/examples/pysp/networkflow
SourceDir="/home/woodruff/software/coopr/src/coopr.pysp/coopr/pysp"
if [ "$1" == "" ]; then
   solver='gurobi'
else
   solver=$1
fi

echo "run3.sh SourceDir is $SourceDir" 
python $SourceDir/lagrangeParam.py -m CCmodels -i CCdata/notuniform10 --solver=$solver --csvPrefix="10" >> run10.out

