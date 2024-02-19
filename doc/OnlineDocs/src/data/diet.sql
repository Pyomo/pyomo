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

INSERT INTO Food VALUES ("Cheeseburger", 1.84, NULL, NULL), ("Ham Sandwich", 2.19, NULL, NULL), ("Hamburger", 1.84, NULL, NULL), ("Fish Sandwich", 1.44, NULL, NULL), ("Chicken Sandwich", 2.29, NULL, NULL), ("Fries", 0.77, NULL, NULL), ("Sausage Biscuit", 1.29, NULL, NULL), ("Lowfat Milk", 0.60, NULL, NULL), ("Orange Juice", 0.72, NULL, NULL);

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
    ('Iron','Orange Juice','2');
