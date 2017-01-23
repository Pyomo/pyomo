#!/bin/sh

pyomo solve --solver=glpk AbstractHLinear.py AbstractH.dat
cat results.yml
rm -f results.yml results.json
