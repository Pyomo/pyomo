#!/bin/sh

pyomo solve --solver=glpk ConcreteHLinear.py 
cat results.yml
