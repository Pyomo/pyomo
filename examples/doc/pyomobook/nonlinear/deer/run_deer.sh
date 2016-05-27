#!/bin/sh

pyomo solve --solver=ipopt --summary DeerProblem.py DeerProblem.dat
