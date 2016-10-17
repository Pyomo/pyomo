#!/bin/sh

# @cmd:
pyomo solve --solver=glpk --postprocess postprocess_fn.py concrete1.py
# @:cmd
cat results.yml
cat results.csv
