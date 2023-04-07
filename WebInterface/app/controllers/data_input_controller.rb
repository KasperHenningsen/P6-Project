class DataInputController < ApplicationController

  def index
    data_option = params[:data_option]
    models = params[:models]

    # TODO: fix params[:models] adds an empty string param
    # puts "models: #{models}"

    case data_option
    when 'manual'
      redirect_to url_for(controller: :data_input, action: :manual_data, models: models)
    when 'api'
      redirect_to redirect_to url_for(controller: :data_input, action: :api_data, models: models)
    when 'upload'
      redirect_to redirect_to url_for(controller: :data_input, action: :upload_data, models: models)
    else
      redirect_to '/'
    end
  end

  def manual_data
    models = params[:models]
  end

  def api_data
    models = params[:models]
  end

  def upload_data
    models = params[:models]
  end
end
