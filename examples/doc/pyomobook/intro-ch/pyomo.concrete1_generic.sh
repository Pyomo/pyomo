#!/bin/sh

pyomo solve --solver=glpk concrete1_generic.py
cat results.yml
