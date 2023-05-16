class SettingsController < ApplicationController

  def new
    @dates = get_date_range
    @horizons = [12, 24, 48]
    @models = get_nn_models

    @setting = Setting.new
  end

  def create
    @setting = Setting.create!(setting_params)
    dataset = Dataset.create!(setting_id: @setting.id)
    start_date = @setting.start_date.iso8601
    end_date = @setting.end_date.iso8601

    @setting.datasets_id = dataset.id

    if @setting.save!
      dataset.save!
      redirect_to profile_path
    else
      @dates = get_date_range
      @horizons = [12, 24, 48]
      @models = get_nn_models
      flash[:error] = "There was an error saving the setting."

      render 'new'
    end

    ActualValueJob.perform_async(dataset.id, start_date, end_date)
    ModelPredictionJob.perform_async(@setting.id, dataset.id)
    ModelLogJob.perform_async(@setting.id)
  end

  def destroy
    setting = Setting.find(params[:id])

    if current_user.id == setting.user_id
      begin
        setting.destroy!
        flash[:success] = "Setting removed!"
      rescue => e
        flash[:error] = "Setting could not be removed! Error: #{e.message}"
      end
    end

    redirect_to profile_path
  end

  private

  def setting_params
    params.require(:setting).permit(:user_id, :start_date, :end_date, :horizon, models: [])
  end

  def get_nn_models
    %w[MLP RNN LSTM GRU CNN TCN Transformer MTGNN]
  end

  def get_date_range
    %w[2000-01-01T00:00:00Z, 2022-01-01T00:00:00Z]
  end
end
