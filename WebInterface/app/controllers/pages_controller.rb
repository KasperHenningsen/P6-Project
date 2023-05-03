require 'async'
require 'async/http/internet'

class PagesController < ApplicationController
  def home
  end

  def spinner
    setting = Setting.find(params[:id])
    $tasks = setting.models.split(',').map do |model|
      ModelPredictionJob.perform_later(model, setting.horizon, setting.start_date, setting.end_date)
    end

    redirect_to graph_path
  end

  def graph
    Async do
      $tasks.each do |task|
        response = task&.wait&.result
        puts "RESPONSE: #{response}"

        if response.nil?
          flash[:error] = "An error occurred while contacting the Models API!"
          redirect_to root_path and return
        else
          @responses << response
        end
      end
      render :graph
    end
  end

  private

  Async def get_model_predictions_async(setting)
    models = setting.models.split(",")
    start_date = setting.start_date.strftime('%Y-%m-%dT%H:%M:%S.%L%z')
    end_date = setting.end_date.strftime('%Y-%m-%dT%H:%M:%S.%L%z')
    tasks = []

    models.each do |model|
      tasks << ModelPredictionJob.perform_later(model, setting.horizon, start_date, end_date)
    end

    tasks
  end
end
