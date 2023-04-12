require 'Time'

class DataInputController < ApplicationController
  def index
    data_option = params[:data_option]
    $start_date = params[:start_date]
    $end_date = params[:end_date]

    case data_option
    when 'manual'
      redirect_to url_for(controller: :data_input, action: :manual)
    when 'api'
      redirect_to url_for(controller: :data_input, action: :api)
    when 'csv'
      redirect_to url_for(controller: :data_input, action: :csv)
    else
      redirect_to root_path
    end
  end

  def manual
    @date = $start_date
    @fields = [['temp', 'temp-input'],
               ['dew_point', 'dew-point-input'],
               ['pressure', 'pressure-input'],
               ['humidity', 'humidity-input'],
               ['rain-one-hour', 'rain-one-hour-input'],
               ['snow-one-hour', 'snow-one-hour-input']]
  end

  def api
    # Call api with date range
    # Then redirect the user

    # https://openweathermap.org/api
    # Filename: hourly1h_16_mmddyy_hhmm.json.gz
    # https://bulk.openweathermap.org/archive/[Filename]?appid=[API_Key]
  end

  def csv
  end

  def upload
  end
end
