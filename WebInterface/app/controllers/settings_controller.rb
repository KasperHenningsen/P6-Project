class SettingsController < ApplicationController

  def new
    @dates = get_date_range
    @horizons = get_horizons
    @models = get_nn_models

    @setting = Setting.new
  end

  def create
    @setting = Setting.create!(setting_params)
    @setting.user_id = current_user.id
    dataset = Dataset.create!(setting_id: @setting.id)
    start_date = @setting.start_date.iso8601
    end_date = @setting.end_date.iso8601

    if @setting.save!
      dataset.save!
      redirect_to profile_path
    else
      @dates = get_date_range
      @horizons = get_horizons
      @models = get_nn_models
      flash[:error] = "There was an error saving the configuration."

      render 'new'
    end

    ActualValueJob.perform_async(@setting.id, start_date, end_date)
    ModelPredictionJob.perform_async(@setting.id)
    ModelLogJob.perform_async(@setting.id)
  end

  def destroy
    setting = Setting.find(params[:id])

    begin
      setting.destroy!
      flash[:success] = "Configuration removed!"
    rescue => e
      puts "ERROR: #{e.message}"
      flash[:error] = "Configuration could not be removed! Error: #{e.message}"
    end

    redirect_to profile_path
  end

  private

  def setting_params
    params.require(:setting).permit(:start_date, :end_date, :horizon, models: [])
  end

  def get_nn_models
    %w[MLP RNN LSTM GRU CNN TCN Transformer MTGNN]
  end

  def get_date_range
    %w[2000-01-01T00:00:00Z, 2022-01-01T00:00:00Z]
  end

  def get_horizons
    [12, 24, 48]
  end
end
