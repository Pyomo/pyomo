#!/usr/bin/env ruby

['config', 'lib'].each do |r|
  require File.join(File.dirname(__FILE__), "#{r}.rb")
end

# Figure out usable files
files = 1.upto(N).map { |i| "Scenario#{i}.dat" } + UserFiles
info("Using files:", 1)
files.each { |f| info f }
info_indent(-1)

# Save existing output file, if any
if File.exist? OutFilename
  File.rename(OutFilename, OutFilename + ".orig")
  info "Moving existing output file to #{OutFilename}.orig"
end

# Header
info("Beginning SQL translation for #{files.length} files...", indent = 1)
output "--"
output "-- MySQL data file for #{ProblemName}"
output "-- Generated #{Time.now}"
output "--"
output

# Drop existing tables
info "Setting up SQL file"
['Demand', 'FCost', 'b0Cost', 'CapCost', 'Arcs', 'Nodes'].each do |table|
  output "drop table if exists #{table};"
end
output

# Create Nodes table
info "Parsing Node information"
output "create table Nodes ( name varchar(16) not null default '', primary key (name) ) engine=innodb;"

# Get node set
set_output = `grep 'set Nodes' #{files.join " "} | cut -f2- -d: | uniq`.chomp.split("\n")
raise "Node names are not consistent across models" if set_output.length > 1

m = /set Nodes := (.*);/.match(set_output[0])
raise "Node set was not recognized" if m.nil?

# Populate Nodes table
node_names = m[1].split
node_count = node_names.length
output "insert into Nodes values #{node_names.to_sql(ParenType::ITEM)};"
output

# Create Arcs table
info "Parsing Arc information"
output "create table Arcs ( 
        first varchar(16) not null, 
        second varchar(16) not null, 
        primary key (first, second),
        foreign key (first) references Nodes (name),
        foreign key (second) references Nodes (name)
        ) engine=innodb ;"

# Get arc set
set_output = `grep 'set Arcs' #{files.join " "} | cut -f2- -d: | uniq`.chomp.split("\n")
raise "Arc pairs are not consistent across models" if set_output.length > 1

m = /set Arcs := (.*);/.match(set_output[0])
raise "Arc set was not recognized" if m.nil?

# Populate Arcs table
arc_names = m[1].split.map { |i| [i[1].chr, i[3].chr] }
arc_count = arc_names.length
output "insert into Arcs values #{arc_names.to_sql(ParenType::NONE)};"
output

# Create CapCost table
info "Parsing CapCost information"
output "create table CapCost (
        first varchar(16) not null,
        second varchar(16) not null,
        cost int not null,
        primary key (first, second),
        foreign key (first, second) references Arcs (first, second)
        ) engine=innodb ;"

# Find CapCost data for all referenced files
capcost_data = files.map { |f| `grep -A#{arc_count} CapCost #{f} | tail -#{arc_count}` }
1.upto(capcost_data.length - 2).each do |i|
  raise "CapCost data differs between models #{files[i]} and #{files[i + 1]}" if capcost_data[i] != capcost_data[i + 1]
end

# Populate CapCost table
output "insert into CapCost values #{list2array(capcost_data[0]).to_sql(ParenType::NONE)};"
output

# Create b0Cost table
info "Parsing b0Cost information"
output "create table b0Cost (
        first varchar(16) not null,
        second varchar(16) not null,
        cost int not null,
        primary key (first, second),
        foreign key (first, second) references Arcs (first, second)
        ) engine=innodb ;"

# Get b0Cost data
b0cost_data = files.map { |f| `grep -A#{arc_count} b0Cost #{f} | tail -#{arc_count}` }
1.upto(b0cost_data.length - 2).each do |i|
  raise "b0Cost data differs between models #{files[i]} and #{files[i + 1]}" if b0cost_data[i] != b0cost_data[i + 1]
end

# Populate b0Cost table
output "insert into b0Cost values #{list2array(b0cost_data[0]).to_sql(ParenType::NONE)};"
output

# Create FCost table
info "Parsing FCost information"
output "create table FCost (
        scenario int not null default 0,
        first varchar(16) not null,
        second varchar(16) not null,
        cost float not null,
        primary key (scenario, first, second),
        foreign key (first) references Nodes (name),
        foreign key (second) references Nodes (name)
        ) engine=innodb ;"

# Get FCost data tables
fcost_data = files.map { |f| `grep -A#{node_count + 1} FCost #{f} | tail -#{node_count + 1}` }

# Populate FCost table
0.upto(files.length - 1).each do |i|
  fd = fcost_data[i]
  pfx = [scenario_number(files[i])]
  output "insert into FCost values #{table2array(fd, prefixes=pfx).to_sql(ParenType::NONE)};"
end
output

# Create Demand table
info "Parsing Demand information"
output "create table Demand (
        scenario int not null default 0,
        first varchar(16) not null,
        second varchar(16) not null,
        demand float not null,
        primary key (scenario, first, second),
        foreign key (first) references Nodes (name),
        foreign key (second) references Nodes (name)
        ) engine=innodb ;"

# Get Demand data
demand_data = files.map { |f| `grep -A#{node_count + 1} Demand #{f} | tail -#{node_count + 1}` }
0.upto(files.length - 1).each do |i|
  dd = demand_data[i]
  pfx = [scenario_number(files[i])]
  output "insert into Demand values #{table2array(dd, prefixes=pfx).to_sql(ParenType::NONE)};"
end
output

# Finish up
info_indent(-1)
info "Done!"
