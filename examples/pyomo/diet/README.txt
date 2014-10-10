========================================
Using Pyomo on the classic diet problem.
========================================

The goal of this problem is to minimize cost while ensuring that the diet
meets certain requirements. To fully illustrate the capabilities of Pyomo,
two examples are given:

* Diet 1 ensures a rounded meal (entree, side, and drink) for minimal cost.
* Diet 2 maximizes nutritional value for minimal cost.

A common data file is used for both diets; however, the models differ, and
certain styles of model (e.g. using an Access database, written in AMPL) are
only available for a certain diet. These availabilities are listed below:

               | Diet1 | Diet2
-----------------------------
Pyomo / .dat   |   Y   |   Y
 AMPL / .dat   |   N   |   Y
Pyomo / .mdb   |   Y   |   N
Pyomo /  SQL   |   Y   |   N
Pyomo / SQLite |   Y   |   N

======
Diet 1
======

Diet 1 provides a model with certain hardcoded constraints: it requires the
purchase of an entree (a sandwich of some kind), a side (either fries or a
McMuffin), and a drink. It attempts to minimize the cost of purchasing this
full meal.

Optimal function value: 2.81

Running with Pyomo
------------------

Files: diet1.py, diet.dat

Run: pyomo diet1.py diet.dat

Running with Pyomo and Access
-----------------------------

Files: diet1.py, diet.db.dat, diet.mdb

Run: pyomo diet1.py diet.db.dat

Notes: The diet.db.dat file is a Pyomo data command file that tells Pyomo
       to import data from the Access database stored in diet.mdb.

Running with Pyomo and MySQL
----------------------------

Files: diet1.py, diet.sql, diet1.sql.dat

Run: mysql -D diet < diet.sql
     pyomo diet1.py diet1.sql.dat

Notes: The diet.sql file is a copy of diet.dat suitable for importing data
       into a MySQL server. It will create the relevant tables; you must
       provide access for the account 'pyomo'@'localhost' (with password
       'pyomopass') to the 'diet' database. In addition, your MySQL account
       must have the ability to create tables in the 'diet' database, and
       to insert values into those tables.

Running with Pyomo and SQLite
-----------------------------

Files: diet1.py, diet.sqlite, diet1.sqlite.dat

Run: pyomo diet1.py diet1.sqlite.dat

Notes: The diet.sqlite file is a copy of diet.dat in the SQLite 3 format. It
       contains identical data to the original .dat file and can be used
       as-is with the built-in Python sqlite3 module. You must be running
       Python 2.5 or newer, since sqlite3 was introduced into the Python
       standard library with that version.

======
Diet 2
======

Diet 2 uses additional data about the nutritional value of each menu item
in order to find the meal with the best nutrition while still minimizing
cost.

Optimal function value: 15.05

Running with AMPL
-----------------

Files:

  diet.dat
  diet2.mod     - The AMPL model
  diet2.ampl    - An AMPL script to solve the diet1 problem
                  (This uses PICO, but other solvers could be used that
                  can read in AMPL *.nl files.)

Run: ampl diet2.ampl

Notes: The directory containing the `cplex` executable must be in your
       system's PATH environment variable.

Running with Pyomo
------------------

Files: diet2.py, diet.dat

Run: pyomo diet2.py diet.dat
