#!/usr/bin/env ruby

require 'optparse'

prefixes = []
suffixes = []
OptionParser.new do |opts|
  opts.banner = "Usage: #{$FILENAME} [options]"

  opts.on("-p", "--prepend VAL", "Prepend a value to tuples. Repeatable.") do |p|
    prefixes.push p
  end

  opts.on("-a", "--append VAL", "Append a value to tuples. Repeatable") do |a|
    suffixes.push a
  end
end.parse!

# Helper
class String
  def smart_split
    sa = self.split(/"/).collect { |x| x.strip }
    return (1..sa.length).zip(sa).collect { |i,x| (i&1).zero? ? x : x.split }.flatten
  end
end

# Get first line for set def
firsts = $stdin.gets.smart_split[0..-2]

# Loop for values
while line = $stdin.gets
  values = line.smart_split
  second = values.shift
  if values[-1] == ';'
    _ = values.slice!(-1)
  end

  if values.length != firsts.length
    raise "Poorly formatted table"
  end

  0.upto(firsts.length - 1).each do |i|
    elements = [prefixes, firsts[i], second, values[i], suffixes].flatten
    puts "(" + elements.map {|i| "'#{i}'"}.join(',') + "),"
  end
end
