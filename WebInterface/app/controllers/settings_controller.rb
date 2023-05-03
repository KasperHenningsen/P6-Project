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

    if @setting.save!
      redirect_to profile_path(id: current_user.id)
    else
      @dates = get_date_range
      @horizons = get_horizons
      @models = get_nn_models
      flash[:error] = "There was an error saving the setting."

      render 'new'
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

  def get_data_methods
    return [["From CSV", "csv"], ["Manual entry", "manual"]]
  end

  def get_nn_models
    return %w[mlp rnn gru lstm mtgnn]
  end

  def get_horizons
    return [12, 16, 24]
  end

  def get_date_range
    return %w[2000-01-01T00:00:00Z, 2022-01-01T00:00:00Z]
  end
end
