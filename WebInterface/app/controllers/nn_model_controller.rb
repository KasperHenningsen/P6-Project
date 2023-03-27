require 'csv'

class NnModelController < ApplicationController
  def index
    #get_data
    hopeThisDosentWork
  end

  def get_data
    csv_text = File.read(Rails.root.join('db', 'open-weather-aalborg-2000-2022.csv'))
    csv = CSV.parse(csv_text, headers: true)

    graph_data = []
    csv.each do |row|
      date_string = Time.at(row['dt'].to_i).to_date.strftime("%d-%m-%Y")
      temp_min = row['temp_min'].to_f
      temp_max = row['temp_max'].to_f
      graph_data << [date_string, temp_min, temp_max]
    end

    10.times do |i|
      print "Row #{i}: [#{graph_data[i][0]}, #{graph_data[i][1]}, #{graph_data[i][2]}]\n"
    end

    @data = graph_data
  end

  def hopeThisDosentWork
    #@db_data = Temperature.all.map { |t| [t.date.to_i, t.temp_min, t.temp_max] }
    @db_data = [[0,2,3],[1,7,8],[2,5,6],[3,1,2],[4,-2,-1],[5,4,5],[6,6,7],[7,12,13],[8,15,16],[9,11,12],[10,9,10]]
  end
end
