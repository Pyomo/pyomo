#!/bin/sh
pyomo solve --solver=glpk diet1.py diet.dat
cat results.yml
