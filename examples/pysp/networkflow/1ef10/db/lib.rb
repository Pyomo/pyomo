class ParenType
  NONE = 0
  ITEM = 1
  ARRAY = 2
end

class Array
  def to_sql(parens = ParenType::ARRAY)
    if parens == ParenType::ARRAY
      return "(" + self.map { |item| item.to_sql }.join(',') + ")"
    elsif parens == ParenType::ITEM
      return self.map { |item| "(#{item.to_sql})" }.join(',')
    elsif parens == ParenType::NONE
      return self.map { |item| item.to_sql }.join(',')
    else
      return self.to_s
    end
  end
end

class String
  def to_sql
    return "'#{self}'"
  end
end

class Object
  def to_sql
    return self.to_s.to_sql
  end
end

def list2array(data)
  result = []
  data.split("\n").each do |line|
    result.push line.split
  end
  return result
end

def table2array(data, prefixes = [], suffixes = [])
  lines = data.split("\n")
  
  # Get first line for set def
  seconds = lines.shift.split[0..-2]
  
  # Loop for values
  result = []
  lines.each do |line|
    # Extract values
    values = line.split
    first = values.shift

    # Drop punctuation
    if values[-1] == ';'
      values.slice!(-1)
    end
  
    # Check lengths match
    if values.length != seconds.length
      raise "Poorly formatted table"
    end
    
    # Actually format tuple
    0.upto(seconds.length - 1).each do |i|
      result.push [prefixes, first, seconds[i], values[i], suffixes].flatten
    end
  end
  return result
end

def scenario_number(filename)
  filename = filename.chomp(".dat")
  m = /Scenario(\d+)/.match(filename)
  if UserMappings.has_key? filename
    return UserMappings[filename]
  elsif not m.nil?
    return m[1].to_i
  else
    return -1
  end
end

$_info_indent = 0
def info(msg, indent = 0)
  puts "[INFO] #{'  ' * $_info_indent}#{msg}"
  info_indent(indent)
end

def info_indent(indent = 0)
  $_info_indent += indent
end

def output(msg = "")
  File.open(OutFilename, "a") { |f| f.puts msg }
end
