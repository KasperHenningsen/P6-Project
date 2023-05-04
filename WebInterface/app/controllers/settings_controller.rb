class SettingsController < ApplicationController

  # Setting
  # - PK
  # - StartDate (datetime)
  # - EndDate (datetime)
  # - Horizon (integer)
  # - Model (string)

  # Prediction
  # - FK ( Ref. Setting)
  # - PK
  # - Temp (float)
  # - Date (datetime)

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
    start_date = to_iso(@setting.start_date)
    end_date = to_iso(@setting.end_date)

    if @setting.save!
      redirect_to profile_path(id: current_user.id)
    else
      @dates = get_date_range
      @horizons = get_horizons
      @models = get_nn_models
      flash[:error] = "There was an error saving the setting."

      render 'new'
    end

    @setting.models.split(',').map do |model|
      ModelPredictionJob.perform_async(model, @setting.horizon, start_date, end_date)
    end
  end

  def delete
    setting = Setting.find(params[:id])

    if current_user.id == setting.user_id
      begin
        setting.destroy!
        flash[:error] = "Setting removed!}"
      rescue => e
        flash[:error] = "Setting could not be removed! Error: #{e.message}"
      end
    end

  end

  private

  def setting_params
    params.require(:setting).permit(:user_id, :start_date, :end_date, :horizon, models: [])
  end

  # @todo: move to SidekiqJob
  def get_data_methods
    return [["From CSV", "csv"], ["Manual entry", "manual"]]
  end

  # @todo: move to SidekiqJob
  def get_nn_models
    return %w[MLP RNN LSTM GRU CNN TCN Transformer MTGNN]
  end

  # @todo: move to SidekiqJob
  def get_horizons
    return [12, 24, 48]
  end

  # @todo: move to SidekiqJob
  def get_date_range
    return %w[2000-01-01T00:00:00Z, 2022-01-01T00:00:00Z]
  end

  def to_iso(date)
    date.strftime('%Y-%m-%dT%H:%M:%S.%L%z')
  end
end
