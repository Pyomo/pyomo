This directory contains scripts and information meant to help ease the
transition of a network flow model into a MySQL database, with all the
data transfer and .dat rewrites that implies. This document describes
the transition process.

## Executive Summary

Get Ruby 1.8, MySQL 5.1. Run `db/gen.rb` to build `1ef10.sql`. Import
that file into your MySQL server. Run `db/switch.rb` to change the
relevant `.dat` files. Re-run the 1ef10 model to verify it works.

## Prerequisites

In order to successfully complete the transition, your system needs:

* The ability to run the network flow model in the first place. (This
  implies Python 2.5 - 2.7, a working Pyomo install, the various
  required modules, etc. See [the Pyomo site][pyomo_install] for
  details on installation.)
* Ruby 1.8. The preferred version is 1.8.7, but most 1.8 series
  interpreters should work. Ruby 1.9 is untested.
* An accessible MySQL server with a spare database. The translation
  scripts are not particularly friendly, flexible, or portable, and so
  it is recommended that you dedicate an entire database to the
  model, or be prepared to make manual modifications throughout the
  process. Recommended version is 5.1; any newer MySQL should suffice.
* The InnoDB MySQL database engine. The network flow models rely on
  in-database foreign key constraints, and the default MyISAM engine
  does not support them. Using InnoDB is necessary for the generated
  SQL files to import.
* An ODBC configuration that supports MySQL connections. On Windows,
  this means you must download a custom MySQL ODBC driver; on Linux
  and similar Unix-based systems, you will need the `unixODBC` package
  and a MySQL connector.
* The `pyodbc` package in a directory on your Python path. Either your
  local Pyomo installation or your systemwide Python installation are
  acceptable.

This guide focuses on Unix-based systems from this point on; adapting
commands to Windows platforms is an exercise left to the reader.

## Step 1: Generating SQL

Before the model can pull from an SQL database, that database must be
populated with the data from the various scenario `.dat` files. Move
to the root directory of the model instance (`1ef10`, in this case)
and open the file `db/gen.rb`. Ensure that the user-configurable values
are correct, then close the file and run:

    $ ruby db/gen.rb

This will create a file called `1ef10.sql` in the instance directory,
backing up and replacing any existing file of the same name. This new
SQL file contains all the relevant statements to create the data tables
and populate them with data extracted from the `.dat` files.

## Step 2: Importing SQL

Once the `1ef10.sql` file has been successfully generated, locate the
database name and credentials for your MySQL server. The scenario will
use the following table names:

* Nodes
* Arcs
* CapCost
* b0Cost
* FCost
* Demand

Your database must have these six tables available; they will be
dropped and recreated on SQL import. The importer does not currently
have SQL table name prefixing available.

From the instance directory, run:

    $ cat 1ef10.sql | mysql -h<HOST> -u<USER> -p -D <DATABASE>

Replace the relevant information as necessary. You will be prompted
for your MySQL password on import; if your account has no password,
drop the `-p` option. The import process should take a few seconds
at most.

Once the import completes, verify the data is stored in the MySQL
database. Run:

    $ mysql -h<HOST> -u<USER> -p
    > USE <DATABASE>;
    > SELECT COUNT (*) FROM Demand;

The Demand table should contain 1100 rows; if it does not, the
import was not successful.

## Step 3: Converting .dat files

Now that the data has been imported into the SQL database, it can
be used to replace the data in .dat files locally. Run:

    $ ruby db/switch.rb

You will be prompted for credentials to your MySQL server, including
your username and password. This password will be written in plaintext -
**do not use a sensitive or valuable password**.

The script will back up the existing `.dat` files in a directory
named `orig`, or a numbered variant if `orig` already exists. To go
back to a file-based network flow problem, simply restore these files.

## Step 4: Running the model

At this point, the database conversion is complete. Run the model to
verify its accuracy, and to check that your ODBC configuration is
working properly.
