#!/bin/bash
runph --user-defined-extension=coopr.pysp.plugins.phhistoryextension --max-iterations=50 --default-rho=1
python ../plot_history.py ph_history.json 
