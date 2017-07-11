#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
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

c.execute('''
CREATE TABLE Food (
    FOOD text not null,
    cost float not null,
    f_min float,
    f_max float,
    PRIMARY KEY (FOOD)
)
''')
conn.commit()

Food_data = [
    ("Cheeseburger",        1.84, None, None),
    ("Ham Sandwich",        2.19, None, None),
    ("Hamburger",           1.84, None, None),
    ("Fish Sandwich",       1.44, None, None),
    ("Chicken Sandwich",    2.29, None, None),
    ("Fries",               0.77, None, None),
    ("Sausage Biscuit",     1.29, None, None),
    ("Lowfat Milk",         0.60, None, None),
    ("Orange Juice",        0.72, None, None)
]
for row in Food_data:
    c.execute('''INSERT INTO Food VALUES (?,?,?,?)''', row)
conn.commit()

c.execute('''
CREATE TABLE Nutr (
    NUTR text not null,
    n_min float,
    n_max float,
    PRIMARY KEY (NUTR)
)
''')

Nutr_data = [
    ("Cal", 2000.0, None),
    ("Carbo", 350.0, 375.0),
    ("Protein", 55.0, None),
    ("VitA", 100.0, None),
    ("VitC", 100.0, None),
    ("Calc", 100.0, None),
    ("Iron", 100.0, None)
]
for row in Nutr_data:
    c.execute('''INSERT INTO Nutr VALUES (?,?,?)''', row)
conn.commit()

c.execute('''
CREATE TABLE Amount (
NUTR text not null,
FOOD varchar not null,
amt float not null,
PRIMARY KEY (NUTR, FOOD),
FOREIGN KEY (NUTR) REFERENCES Nutr (NUTR),
FOREIGN KEY (FOOD) REFERENCES Food (FOOD)
)
''')
conn.commit()

Amount_data = [
    ('Cal','Cheeseburger','510'),
    ('Carbo','Cheeseburger','34'),
    ('Protein','Cheeseburger','28'),
    ('VitA','Cheeseburger','15'),
    ('VitC','Cheeseburger','6'),
    ('Calc','Cheeseburger','30'),
    ('Iron','Cheeseburger','20'),
    ('Cal','Ham Sandwich','370'),
    ('Carbo','Ham Sandwich','35'),
    ('Protein','Ham Sandwich','24'),
    ('VitA','Ham Sandwich','15'),
    ('VitC','Ham Sandwich','10'),
    ('Calc','Ham Sandwich','20'),
    ('Iron','Ham Sandwich','20'),
    ('Cal','Hamburger','500'),
    ('Carbo','Hamburger','42'),
    ('Protein','Hamburger','25'),
    ('VitA','Hamburger','6'),
    ('VitC','Hamburger','2'),
    ('Calc','Hamburger','25'),
    ('Iron','Hamburger','20'),
    ('Cal','Fish Sandwich','370'),
    ('Carbo','Fish Sandwich','38'),
    ('Protein','Fish Sandwich','14'),
    ('VitA','Fish Sandwich','2'),
    ('VitC','Fish Sandwich','0'),
    ('Calc','Fish Sandwich','15'),
    ('Iron','Fish Sandwich','10'),
    ('Cal','Chicken Sandwich','400'),
    ('Carbo','Chicken Sandwich','42'),
    ('Protein','Chicken Sandwich','31'),
    ('VitA','Chicken Sandwich','8'),
    ('VitC','Chicken Sandwich','15'),
    ('Calc','Chicken Sandwich','15'),
    ('Iron','Chicken Sandwich','8'),
    ('Cal','Fries','220'),
    ('Carbo','Fries','26'),
    ('Protein','Fries','3'),
    ('VitA','Fries','0'),
    ('VitC','Fries','15'),
    ('Calc','Fries','0'),
    ('Iron','Fries','2'),
    ('Cal','Sausage Biscuit','345'),
    ('Carbo','Sausage Biscuit','27'),
    ('Protein','Sausage Biscuit','15'),
    ('VitA','Sausage Biscuit','4'),
    ('VitC','Sausage Biscuit','0'),
    ('Calc','Sausage Biscuit','20'),
    ('Iron','Sausage Biscuit','15'),
    ('Cal','Lowfat Milk','110'),
    ('Carbo','Lowfat Milk','12'),
    ('Protein','Lowfat Milk','9'),
    ('VitA','Lowfat Milk','10'),
    ('VitC','Lowfat Milk','4'),
    ('Calc','Lowfat Milk','30'),
    ('Iron','Lowfat Milk','0'),
    ('Cal','Orange Juice','80'),
    ('Carbo','Orange Juice','20'),
    ('Protein','Orange Juice','1'),
    ('VitA','Orange Juice','2'),
    ('VitC','Orange Juice','120'),
    ('Calc','Orange Juice','2'),
    ('Iron','Orange Juice','2')
]
for row in Amount_data:
    c.execute('''INSERT INTO Amount VALUES (?,?,?)''', row)
conn.commit()
