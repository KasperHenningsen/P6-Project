require('net/http')

class GraphController < ApplicationController
  def show
    setting = Setting.find(params[:id])
    start_date_iso = setting.start_date.iso8601
    end_date_iso = setting.end_date.iso8601

    actuals = ActualValuesJob.perform_async(start_date_iso, end_date_iso)
    unless actuals == nil
      @dates = actuals.dates
      @datasets = [actuals]
      setting.models.split(',').each do |model|
        pred = ModelPredictionJob.perform_sync(model, setting.horizon, start_date_iso, end_date_iso)
        unless pred == nil
          @datasets.append(pred)
        end
      end
    end
    render('pages/graph')
  end
end
