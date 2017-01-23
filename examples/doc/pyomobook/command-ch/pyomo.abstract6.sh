#!/bin/sh

# @cmd:
pyomo solve --solver=glpk --model-name=Model \
            abstract6.py abstract6.dat
# @:cmd
cat results.yml
rm -f results.yml results.json
