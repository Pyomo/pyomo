#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

"""
Create the PP.sqlite file
"""

import sqlite3

conn = sqlite3.connect('PP.sqlite')

c = conn.cursor()

for table in ['PPtable']:
    c.execute('DROP TABLE IF EXISTS ' + table)
conn.commit()

c.execute('''
CREATE TABLE PPtable (
    A text not null,
    B text not null,
    PP float not null
)
''')
conn.commit()

data = [
    ("A1", "B1", 4.3),
    ("A2", "B2", 4.4),
    ("A3", "B3", 4.5)
]
for row in data:
    c.execute('''INSERT INTO PPtable VALUES (?,?,?)''', row)
conn.commit()
