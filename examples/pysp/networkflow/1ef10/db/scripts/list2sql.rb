#!/usr/bin/env ruby

while line = $stdin.gets
  puts "(" + line.split.map {|i| "'#{i}'"}.join(',') + "),"
end
