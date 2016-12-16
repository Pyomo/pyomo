#!/bin/sh

pyomo solve --solver=glpk MiscAbstract2.py MiscAbstract.dat > /dev/null 2>&1
cat results.yml
