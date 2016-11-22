#!/bin/sh

pyomo solve --solver=glpk abstract4.py
cat results.yml
