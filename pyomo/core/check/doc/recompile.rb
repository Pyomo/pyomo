#!/usr/bin/env ruby

require 'rb-inotify'

notifier = INotify::Notifier.new

notifier.watch(".", :modify) do |event|
  if event.name.end_with? ".tex"
    puts "#{`date`.chomp}: Modified #{event.name}; recompiling..."
    `pdflatex #{event.name} > /dev/null 2>&1`
  end
end

notifier.run
