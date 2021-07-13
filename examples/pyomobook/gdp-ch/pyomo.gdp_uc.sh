#!/bin/sh

# @cmd:
pyomo solve gdp_uc.py gdp_uc.dat --solver=glpk \
    --transform core.logical_to_linear --transform gdp.bigm
# @:cmd
cat results.yml
rm results.yml
