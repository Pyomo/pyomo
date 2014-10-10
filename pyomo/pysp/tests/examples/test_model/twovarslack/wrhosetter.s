#!/bin/bash
runph --user-defined-extension=pyomo.pysp.plugins.phhistoryextension --max-iterations=50 --rho-cfgfile=rhosetter.cfg
python ../plot_history.py ph_history.json 
