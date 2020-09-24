#!/bin/sh

pyomo solve --solver=glpk concrete4a.py
cat results.yml
