#!/bin/sh

pyomo solve --solver=glpk concrete4.py
cat results.yml
