#!/bin/sh

pyomo solve --solver=glpk abstract5.py abstract5.dat
cat results.yml
rm -f results.yml results.json
