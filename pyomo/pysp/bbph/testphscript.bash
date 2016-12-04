#! /bin/bash
EDIR=/home/woodruff/software/pyomo/examples/pysp/sizes

export PYRO_NS_PORT=9090
runph -i $EDIR/SIZES3 -m $EDIR/models \
      --default-rho=1 \
      --user-defined-extension=pyomo.pysp.plugins.phboundextension \
      --traceback BBPHTOKEN

