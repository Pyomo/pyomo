#! /bin/bash
# test the bb code

BDIR=/home/woodruff/software/pyomo
EDIR=$BDIR/examples/pysp/sizes

python BBPH.py -i $EDIR/SIZES3 -m $EDIR/models --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-working-dir=workingdir --BBPH-ph-Script=testphscript.bash --BBPH-ph-options-token=BBPHTOKEN --BBPH-Verbose --BBPH-brancher-plugin=/export/home/dlwoodruff/Documents/phbb/code/brancher --BBPH-brancher-output=brancherout.p --BBPH-OuterIterationLimit=4 --BBPH-PH-Launch-Limit=4

#python BBPH.py -i $EDIR/SIZES3 -m $EDIR/models --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-OuterIterationLimit=5 --max-iterations=5 #--pyro-host=localhost --solver-manager=phpyro --phpyro-required-workers=3

#python BBPH.py -i $EDIR -m $EDIR --default-rho=1 --user-defined-extension=pyomo.pysp.plugins.phboundextension --traceback --BBPH-OuterIterationLimit=5 --max-iterations=5 #--pyro-host=localhost --solver-manager=phpyro --phpyro-required-workers=3
