#!/bin/sh

pyomo solve --solver=glpk concrete4b.py
cat results.yml
