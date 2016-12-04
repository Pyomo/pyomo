#! /bin/bash
# test the bb code

EDIR=../../../examples/pysp/sizes
#EDIR=$BASEDIR/pyomo/src/pyomo/examples/pysp/farmerWintegers
#EDIR=$BASEDIR/pyomo/src/pyomo/examples/pysp/networkflow
#EDIR=/export/home/dlwoodruff/software/pyomo/src/pyomo/examples/pysp/sizes
#EDIR=/home/dlwoodruff/data/pyomoexamples/pyomo_examples_1886/pysp/sizes
#EDIR=$BASEDIR/pyomo/src/pyomo/pyomo/pysp/tests/examples/test_model/twovarslack
#EDIR=$BASEDIR/pyomo/src/pyomo/pyomo/pysp/tests/examples/test_model/feas
#EDIR=../impossible

python bbph.py -i $EDIR/SIZES3 -m $EDIR/models --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-Verbose --BBPH-OuterIterationLimit=4 --max-iterations=40 --or-convergers --termdiff-threshold=0.005

#python BBPH.py -i $EDIR/SIZES3 -m $EDIR/models --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-OuterIterationLimit=5 --max-iterations=5 #--pyro-host=localhost --solver-manager=phpyro --phpyro-required-workers=3

#python BBPH.py -i $EDIR -m $EDIR --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-OuterIterationLimit=5 --max-iterations=5 #--pyro-host=localhost --solver-manager=phpyro --phpyro-required-workers=3
