#!/bin/bash
IDIR=../../../examples/pysp/networkflow

python bbph.py -i $IDIR/1ef10 -m $IDIR/models --rho-cfgfile=$IDIR/config/rhosetter0.5 --enable-ww-extensions --ww-extension-cfgfile=$IDIR/config/wwph-fixlag10.cfg --ww-extension-suffixfile=$IDIR/config/wwph.suffixes --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --max-iterations=20 --BBPH-OuterIterationLimit=3 --xhat-method=config/slam.suffixes --or-convergers --termdiff-threshold=0.005

