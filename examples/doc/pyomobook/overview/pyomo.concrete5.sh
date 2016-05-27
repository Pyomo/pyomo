#!/bin/sh

pyomo solve --solver=glpk concrete5.py
cat results.yml
