#!/bin/sh

pyomo solve --solver=glpk --summary ConcreteHLinear.py 
rm -f results.yml results.json

