#!/usr/bin/env ruby

['config', 'lib'].each do |r|
  require File.join(File.dirname(__FILE__), "#{r}.rb")
end

# Get info from user
config = {}

print "MySQL hostname [localhost]: "
config['host'] = gets.chomp
config['host'] = 'localhost' if config['host'].empty?

print "MySQL database [networkflow]: "
config['db'] = gets.chomp
config['db'] = 'networkflow' if config['db'].empty?

print "MySQL user []: "
config['user'] = gets.chomp

print "MySQL password []: "
config['pass'] = gets.chomp

# Figure out usable files
files = 1.upto(N).map { |i| "Scenario#{i}.dat" } + UserFiles

# Save all existing files
if File.exists? "orig"
  existing = [0]
  Dir.glob("orig.*") do |fn|
    puts "Globbing #{fn}"
    m = /^orig\.([0-9]+)$/.match(fn)
    if m
      existing.push m[1].to_i
    end
  end
  backup_dir = "orig.#{existing.sort[-1] + 1}"
else
  backup_dir = "orig"
end
info "Backing up into directory #{backup_dir}"

# Move and rewrite files
info("Replacing #{files.length} .dat files", indent = 1)
Dir.mkdir(backup_dir)
files.each do |fn|
  info "Processing #{fn}..."
  # Back up
  File.rename(fn, File.join(backup_dir, fn))

  # Common data
  data = [
          {:columns => "name", :name => "Nodes", :format => "set"},
          {:columns => "first,second", :name => "Arcs", :format => "set"},
          {:columns => "first,second,cost", :name => "CapCost", :format => "param"},
          {:columns => "first,second,cost", :name => "b0Cost", :format => "param"},
          {:columns => "first,second,cost", :name => "FCost", :format => "param", :where => "scenario"},
          {:columns => "first,second,demand", :name => "Demand", :format => "param", :where => "scenario"}
         ]

  def query(d, sn)
    result = "SELECT #{d[:columns]} FROM #{d[:name]}"
    if ! d[:where].nil?
      result += " WHERE #{d[:where]}=#{sn}"
    end
    return result
  end

  # Rewrite
  File.open(fn, 'w') do |file|
    sn = scenario_number(fn)

    import_def = "Driver={MySQL};Database=#{config['db']};Server=#{config['host']};User=#{config['user']};Password=#{config['pass']};"
    data.each do |d|
      file.puts "import \"#{import_def}\" using=pyodbc query=\"#{query(d, sn)}\" format=#{d[:format]} : #{d[:name]} ;"
    end
  end
end

info_indent(-1)
info "Done!"
