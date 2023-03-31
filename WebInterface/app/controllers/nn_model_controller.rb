require 'date'
require 'async'

class NnModelController < ApplicationController
  def index
    limit = (24 * 31)
    #limit = get_limit(params[:datapoints])

    #puts "Selector: #{params[:datapoints]}"
    #puts "Limit: #{limit}"

    @data = get_data_async(limit)
  end

  def get_limit(option)
    case option
      when "1 Day"
        return 24
      when "7 Days"
        return (24 * 7)
      when "1 Month"
        return (24 * 31)
      when "6 Months"
        return (24 * (365 / 2))
      when "1 Year"
        return (24 * 365)
      when "2 Years"
        return (2 * (24 * 365))
      when "5 Years"
        return (5 * (24 * 365))
      when "10 Years"
        return (10 * (24 * 365))
      else
        return nil # fetch all data points
      end
  end

  def get_data_async(limit)
    Async do
      if limit
        Temperature.order(:date).limit(limit).map { |t| [
          t.date,
          t.temp_min,
          t.temp_max
        ]}
      else
        Temperature.order(:date).map { |t| [
          t.date,
          t.temp_min,
          t.temp_max
        ]}
      end
    end.wait
  end
end
