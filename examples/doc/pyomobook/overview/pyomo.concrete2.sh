#!/bin/sh

pyomo solve --solver=glpk concrete2.py
cat results.yml
