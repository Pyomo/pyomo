#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Create the diet.sqlite file with all the appropriate data.
"""

import sqlite3

conn = sqlite3.connect('diet.sqlite')

c = conn.cursor()

for table in ['Amount', 'Nutr', 'Food']:
    c.execute('DROP TABLE IF EXISTS ' + table)
conn.commit()

c.execute(
    '''
CREATE TABLE Food (
    FOOD text not null,
    cost float not null,
    f_min float,
    f_max float,
    PRIMARY KEY (FOOD)
)
'''
)
conn.commit()

Food_data = [
    ("Quarter Pounder w Cheese", 1.84, None, None),
    ("McLean Deluxe w Cheese", 2.19, None, None),
    ("Big Mac", 1.84, None, None),
    ("Filet-O-Fish", 1.44, None, None),
    ("McGrilled Chicken", 2.29, None, None),
    ("Fries, small", 0.77, None, None),
    ("Sausage McMuffin", 1.29, None, None),
    ("1% Lowfat Milk", 0.60, None, None),
    ("Orange Juice", 0.72, None, None),
]
for row in Food_data:
    c.execute('''INSERT INTO Food VALUES (?,?,?,?)''', row)
conn.commit()

c.execute(
    '''
CREATE TABLE Nutr (
    NUTR text not null,
    n_min float,
    n_max float,
    PRIMARY KEY (NUTR)
)
'''
)

Nutr_data = [
    ("Cal", 2000.0, None),
    ("Carbo", 350.0, 375.0),
    ("Protein", 55.0, None),
    ("VitA", 100.0, None),
    ("VitC", 100.0, None),
    ("Calc", 100.0, None),
    ("Iron", 100.0, None),
]
for row in Nutr_data:
    c.execute('''INSERT INTO Nutr VALUES (?,?,?)''', row)
conn.commit()

c.execute(
    '''
CREATE TABLE Amount (
NUTR text not null,
FOOD varchar not null,
amt float not null,
PRIMARY KEY (NUTR, FOOD),
FOREIGN KEY (NUTR) REFERENCES Nutr (NUTR),
FOREIGN KEY (FOOD) REFERENCES Food (FOOD)
)
'''
)
conn.commit()

Amount_data = [
    ('Cal', 'Quarter Pounder w Cheese', '510'),
    ('Carbo', 'Quarter Pounder w Cheese', '34'),
    ('Protein', 'Quarter Pounder w Cheese', '28'),
    ('VitA', 'Quarter Pounder w Cheese', '15'),
    ('VitC', 'Quarter Pounder w Cheese', '6'),
    ('Calc', 'Quarter Pounder w Cheese', '30'),
    ('Iron', 'Quarter Pounder w Cheese', '20'),
    ('Cal', 'McLean Deluxe w Cheese', '370'),
    ('Carbo', 'McLean Deluxe w Cheese', '35'),
    ('Protein', 'McLean Deluxe w Cheese', '24'),
    ('VitA', 'McLean Deluxe w Cheese', '15'),
    ('VitC', 'McLean Deluxe w Cheese', '10'),
    ('Calc', 'McLean Deluxe w Cheese', '20'),
    ('Iron', 'McLean Deluxe w Cheese', '20'),
    ('Cal', 'Big Mac', '500'),
    ('Carbo', 'Big Mac', '42'),
    ('Protein', 'Big Mac', '25'),
    ('VitA', 'Big Mac', '6'),
    ('VitC', 'Big Mac', '2'),
    ('Calc', 'Big Mac', '25'),
    ('Iron', 'Big Mac', '20'),
    ('Cal', 'Filet-O-Fish', '370'),
    ('Carbo', 'Filet-O-Fish', '38'),
    ('Protein', 'Filet-O-Fish', '14'),
    ('VitA', 'Filet-O-Fish', '2'),
    ('VitC', 'Filet-O-Fish', '0'),
    ('Calc', 'Filet-O-Fish', '15'),
    ('Iron', 'Filet-O-Fish', '10'),
    ('Cal', 'McGrilled Chicken', '400'),
    ('Carbo', 'McGrilled Chicken', '42'),
    ('Protein', 'McGrilled Chicken', '31'),
    ('VitA', 'McGrilled Chicken', '8'),
    ('VitC', 'McGrilled Chicken', '15'),
    ('Calc', 'McGrilled Chicken', '15'),
    ('Iron', 'McGrilled Chicken', '8'),
    ('Cal', 'Fries, small', '220'),
    ('Carbo', 'Fries, small', '26'),
    ('Protein', 'Fries, small', '3'),
    ('VitA', 'Fries, small', '0'),
    ('VitC', 'Fries, small', '15'),
    ('Calc', 'Fries, small', '0'),
    ('Iron', 'Fries, small', '2'),
    ('Cal', 'Sausage McMuffin', '345'),
    ('Carbo', 'Sausage McMuffin', '27'),
    ('Protein', 'Sausage McMuffin', '15'),
    ('VitA', 'Sausage McMuffin', '4'),
    ('VitC', 'Sausage McMuffin', '0'),
    ('Calc', 'Sausage McMuffin', '20'),
    ('Iron', 'Sausage McMuffin', '15'),
    ('Cal', '1% Lowfat Milk', '110'),
    ('Carbo', '1% Lowfat Milk', '12'),
    ('Protein', '1% Lowfat Milk', '9'),
    ('VitA', '1% Lowfat Milk', '10'),
    ('VitC', '1% Lowfat Milk', '4'),
    ('Calc', '1% Lowfat Milk', '30'),
    ('Iron', '1% Lowfat Milk', '0'),
    ('Cal', 'Orange Juice', '80'),
    ('Carbo', 'Orange Juice', '20'),
    ('Protein', 'Orange Juice', '1'),
    ('VitA', 'Orange Juice', '2'),
    ('VitC', 'Orange Juice', '120'),
    ('Calc', 'Orange Juice', '2'),
    ('Iron', 'Orange Juice', '2'),
]
for row in Amount_data:
    c.execute('''INSERT INTO Amount VALUES (?,?,?)''', row)
conn.commit()
