#!/bin/sh

pyomo solve --solver=glpk concrete2a.py
cat results.yml
