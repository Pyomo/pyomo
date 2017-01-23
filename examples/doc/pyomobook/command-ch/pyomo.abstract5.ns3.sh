#!/bin/sh

# @cmd:
pyomo solve --solver=glpk --namespace=c1 --namespace=data2 \
                abstract5.py abstract5-ns2.dat
# @:cmd
cat results.yml
rm -f results.yml results.json
