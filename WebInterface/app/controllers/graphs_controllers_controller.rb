require 'csv'

class GraphsControllersController < ApplicationController

  def index

  end

  def temperature_data
    csv_data = CSV.read(Rails.root.join('db', 'temperature_data.csv'))
    data = csv_data.drop(1).map { |row| [Time.at(row[0].to_i), row[1].to_f, row[2].to_f] }
    render partial: 'temperature_data', locals: { data: data }
  end


  def show
    # Read the data from the CSV file
    csv_text = File.read(Rails.root.join('public', 'data.csv'))
    csv = CSV.parse(csv_text, headers: true)

    # Format the data for the graph
    graph_data = csv.map do |row|
      datetime = Time.at(row['dt'].to_i).strftime("%Y-%m-%d %H:%M:%S")
      temp_min = row['temp_min'].to_f
      temp_max = row['temp_max'].to_f
      [datetime, temp_min, temp_max]
    end

    # Pass the graph data to the view
    @data = graph_data.to_json
  end

  def show_fake
    # Generate test data for the graph
    start_date = Date.new(2023, 3, 1)
    end_date = Date.new(2023, 3, 31)
    graph_data = (start_date..end_date).map do |date|
      temp_min = rand(10..20)
      temp_max = rand(temp_min..30)
      [date, temp_min, temp_max]
    end

    # Pass the graph data to the view
    @fake_data = graph_data.to_json
  end
end
