#!/bin/sh

# @cmd:
pyomo solve gdp_uc.py gdp_uc.dat --transform gdp.bigm --solver=glpk
# @:cmd
cat results.yml
rm results.yml
