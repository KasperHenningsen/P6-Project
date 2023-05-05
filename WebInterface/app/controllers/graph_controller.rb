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

  private

  def get_actuals(start_date, end_date)
    start_date_iso = start_date.iso8601
    end_date_iso = end_date.iso8601
    req_url = URI("http://127.0.0.1:5000/dataset/actuals?start_date=#{start_date_iso}&end_date=#{end_date_iso}")
    dataset = make_request(req_url)
    unless dataset == nil
      dataset.identifier = "Actual"
      return dataset
    end
    return nil
  end

  def get_predictions(model, horizon, start_date, end_date)
    start_date_iso = start_date.iso8601
    end_date_iso = end_date.iso8601
    req_url = URI("http://127.0.0.1:5000/predictions/models/#{model.downcase}?horizon=#{horizon}&start_date=#{start_date_iso}&end_date=#{end_date_iso}")
    dataset = make_request(req_url)
    unless dataset == nil
      dataset.identifier = model.upcase
      return dataset
    end
    return nil
  end

  def make_request(request)
    begin
      res = Net::HTTP.get_response(request)
      if res.code == '200'
        data = JSON.parse(res.body)
        return Dataset.new(
          identifier: 'Unnamed',
          dates: data['dates'].map { |d| DateTime.parse(d).to_fs(:short) if d },
          temps: data['temps'].map { |t| t.to_f.round(2) if t }
        )
      else
        config.logger.warn("retrieved non-OK status from API: #{res.code}")
      end
    rescue => e
      config.logger.warn(e.detailed_message)
    end
    return nil
  end
end
