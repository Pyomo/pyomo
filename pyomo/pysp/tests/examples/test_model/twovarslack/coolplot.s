#!/bin/bash
runph --user-defined-extension=coopr.pysp.plugins.phhistoryextension --max-iterations=50 --default-rho=200 --report-weights --enable-ww-extensions
python ../plot_history.py ph_history.json 
