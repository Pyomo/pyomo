#!/bin/sh

pyomo solve --solver=ipopt --logging=quiet disease_estimation.py disease_estimation.dat
