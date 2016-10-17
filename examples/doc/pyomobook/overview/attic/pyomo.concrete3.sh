#!/bin/sh

pyomo solve --solver=glpk concrete3.py
cat results.yml
