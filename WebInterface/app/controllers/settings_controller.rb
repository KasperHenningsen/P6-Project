class SettingsController < ApplicationController
  before_action :set_setting, only: %i[ show edit update destroy ]

  # GET /settings or /settings.json
  def index
    @settings = Setting.all
  end

  # GET /settings/1 or /settings/1.json
  def show
  end

  # GET /settings/new
  def new
    @dates = get_date_range
    @data_methods = get_data_methods
    @horizons = get_horizons
    @models = get_nn_models

    @setting = Setting.new
  end

  # GET /settings/1/edit
  def edit
  end

  # POST /settings or /settings.json
  def create
    @data_methods = get_data_methods
    @setting = Setting.new(setting_params)

    if @setting.save
      redirect_to pages_load_path
    else
      flash[:error] = "There was an error saving the setting."
      redirect_to new_setting_path
    end
  end

  # PATCH/PUT /settings/1 or /settings/1.json
  def update
    respond_to do |format|
      if @setting.update(setting_params)
        format.html { redirect_to setting_url(@setting), notice: "Setting was successfully updated." }
      else
        format.html { render :edit, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /settings/1 or /settings/1.json
  def destroy
    @setting.destroy

    respond_to do |format|
      format.html { redirect_to settings_url, notice: "Setting was successfully destroyed." }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_setting
      @setting = Setting.find(params[:id])
    end

    # Only allow a list of trusted parameters through.
    def setting_params
      params.require(:setting).permit(:start_date, :end_date, :data_method)
    end

    def get_data_methods
      return [["From CSV", "csv"], ["Manual entry", "manual"]]
    end

    def get_nn_models
      return ["mlp", "rnn", "gru", "lstm", "mtgnn"]
    end

    def get_horizons
      return [12, 16, 24]
    end

    def get_date_range
      return ["2000-01-01T00:00:00Z", "2022-01-01T00:00:00Z"]
    end
end
