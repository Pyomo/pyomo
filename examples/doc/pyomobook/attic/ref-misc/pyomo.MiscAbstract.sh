#!/bin/sh

pyomo solve --solver=glpk MiscAbstract.py MiscAbstract.dat > /dev/null 2>&1
cat results.yml
