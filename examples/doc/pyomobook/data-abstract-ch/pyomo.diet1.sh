#!/bin/sh
pyomo solve --solver=glpk diet1.py diet.sqlite.dat
cat results.yml
