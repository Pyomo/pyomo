DROP TABLE IF EXISTS Amount;
DROP TABLE IF EXISTS Nutr;
DROP TABLE IF EXISTS Food;

CREATE TABLE Food (
    FOOD varchar(64) not null,
    cost float not null,
    f_min float,
    f_max float,
    primary key (FOOD)
    ) engine=innodb ;

INSERT INTO Food VALUES ("Quarter Pounder w Cheese", 1.84, NULL, NULL), ("McLean Deluxe w Cheese", 2.19, NULL, NULL), ("Big Mac", 1.84, NULL, NULL), ("Filet-O-Fish", 1.44, NULL, NULL), ("McGrilled Chicken", 2.29, NULL, NULL), ("Fries, small", 0.77, NULL, NULL), ("Sausage McMuffin", 1.29, NULL, NULL), ("1% Lowfat Milk", 0.60, NULL, NULL), ("Orange Juice", 0.72, NULL, NULL);

CREATE TABLE Nutr (
    NUTR varchar(64) not null,
    n_min float,
    n_max float,
    primary key (NUTR)
    ) engine=innodb ;

INSERT INTO Nutr VALUES ("Cal", 2000.0, NULL), ("Carbo", 350.0, 375.0), ("Protein", 55.0, NULL), ("VitA", 100.0, NULL), ("VitC", 100.0, NULL), ("Calc", 100.0, NULL), ("Iron", 100.0, NULL);

CREATE TABLE Amount (
    NUTR varchar(64) not null,
    FOOD varchar(64) not null,
    amt float not null,
    primary key (NUTR, FOOD),
    foreign key (NUTR) references Nutr (NUTR),
    foreign key (FOOD) references Food (FOOD)
    ) engine=innodb ;

INSERT INTO Amount VALUES
    ('Cal','Quarter Pounder w Cheese','510'),
    ('Carbo','Quarter Pounder w Cheese','34'),
    ('Protein','Quarter Pounder w Cheese','28'),
    ('VitA','Quarter Pounder w Cheese','15'),
    ('VitC','Quarter Pounder w Cheese','6'),
    ('Calc','Quarter Pounder w Cheese','30'),
    ('Iron','Quarter Pounder w Cheese','20'),
    ('Cal','McLean Deluxe w Cheese','370'),
    ('Carbo','McLean Deluxe w Cheese','35'),
    ('Protein','McLean Deluxe w Cheese','24'),
    ('VitA','McLean Deluxe w Cheese','15'),
    ('VitC','McLean Deluxe w Cheese','10'),
    ('Calc','McLean Deluxe w Cheese','20'),
    ('Iron','McLean Deluxe w Cheese','20'),
    ('Cal','Big Mac','500'),
    ('Carbo','Big Mac','42'),
    ('Protein','Big Mac','25'),
    ('VitA','Big Mac','6'),
    ('VitC','Big Mac','2'),
    ('Calc','Big Mac','25'),
    ('Iron','Big Mac','20'),
    ('Cal','Filet-O-Fish','370'),
    ('Carbo','Filet-O-Fish','38'),
    ('Protein','Filet-O-Fish','14'),
    ('VitA','Filet-O-Fish','2'),
    ('VitC','Filet-O-Fish','0'),
    ('Calc','Filet-O-Fish','15'),
    ('Iron','Filet-O-Fish','10'),
    ('Cal','McGrilled Chicken','400'),
    ('Carbo','McGrilled Chicken','42'),
    ('Protein','McGrilled Chicken','31'),
    ('VitA','McGrilled Chicken','8'),
    ('VitC','McGrilled Chicken','15'),
    ('Calc','McGrilled Chicken','15'),
    ('Iron','McGrilled Chicken','8'),
    ('Cal','Fries, small','220'),
    ('Carbo','Fries, small','26'),
    ('Protein','Fries, small','3'),
    ('VitA','Fries, small','0'),
    ('VitC','Fries, small','15'),
    ('Calc','Fries, small','0'),
    ('Iron','Fries, small','2'),
    ('Cal','Sausage McMuffin','345'),
    ('Carbo','Sausage McMuffin','27'),
    ('Protein','Sausage McMuffin','15'),
    ('VitA','Sausage McMuffin','4'),
    ('VitC','Sausage McMuffin','0'),
    ('Calc','Sausage McMuffin','20'),
    ('Iron','Sausage McMuffin','15'),
    ('Cal','1% Lowfat Milk','110'),
    ('Carbo','1% Lowfat Milk','12'),
    ('Protein','1% Lowfat Milk','9'),
    ('VitA','1% Lowfat Milk','10'),
    ('VitC','1% Lowfat Milk','4'),
    ('Calc','1% Lowfat Milk','30'),
    ('Iron','1% Lowfat Milk','0'),
    ('Cal','Orange Juice','80'),
    ('Carbo','Orange Juice','20'),
    ('Protein','Orange Juice','1'),
    ('VitA','Orange Juice','2'),
    ('VitC','Orange Juice','120'),
    ('Calc','Orange Juice','2'),
    ('Iron','Orange Juice','2');
