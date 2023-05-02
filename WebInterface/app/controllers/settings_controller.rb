class SettingsController < ApplicationController

  # GET /settings/new
  def new
    @dates = get_date_range
    @data_methods = get_data_methods
    @horizons = get_horizons
    @models = get_nn_models

    @setting = Setting.new
  end

  def create
    @data_methods = get_data_methods
    @setting = Setting.new(setting_params)

    if @setting.save!
      redirect_to graph_path(id: @setting.id)
    else
      @dates = get_date_range
      @horizons = get_horizons
      @models = get_nn_models
      flash.now[:error] = "There was an error saving the setting."

      render 'new'
    end
  end

  private

  def setting_params
    params.require(:setting).permit(:start_date, :end_date, :horizon, models: [])
  end

  def get_data_methods
    return [["From CSV", "csv"], ["Manual entry", "manual"]]
  end

  def get_nn_models
    return %w[mlp rnn gru lstm mtgnn]
  end

  def get_horizons
    return [12, 24, 48]
  end

  def get_date_range
    return ["2000-01-01T00:00:00Z", "2022-01-01T00:00:00Z"]
  end
end
