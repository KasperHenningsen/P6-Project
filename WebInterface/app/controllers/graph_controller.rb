class GraphController < ApplicationController
  def show
    setting = Setting.find(params[:id])
    @dataset = get_predictions(setting.start_date, setting.end_date)
    @actuals = get_actuals(setting.start_date, setting.end_date)
    render('pages/graph')
  end

  private
  require('net/http')

  def get_actuals(start_date, end_date)
    start_date_iso = start_date.iso8601
    end_date_iso = end_date.iso8601
    req_url = URI("http://127.0.0.1:5000/dataset/actuals?start_date=#{start_date_iso}&end_date=#{end_date_iso}")

    begin
      res = Net::HTTP.get_response(req_url)

      if res.code == '200'
        data = JSON.parse(res.body)
        return Dataset.new(
          identifier: 'Actual',
          dates: data['dates'].map { |d| DateTime.parse(d).to_fs(:short) if d },
          temps: data['temps'].map { |t| t.to_f.round(1) if t }
        )
      else
        config.logger.warn('response was not OK!')
        nil
      end
    rescue => e
      config.logger.warn(e.detailed_message)
      nil
    end
  end

  def get_predictions(start_date, end_date)
    start_date_iso = start_date.iso8601
    end_date_iso = end_date.iso8601
    req_url = URI("http://127.0.0.1:5000/predictions/models/cnn?horizon=12&start_date=#{start_date_iso}&end_date=#{end_date_iso}")

    begin
      res = Net::HTTP.get_response(req_url)

      if res.code == '200'
        data = JSON.parse(res.body)
        return Dataset.new(
          identifier: 'CNN',
          dates: data['dates'].map { |d| DateTime.parse(d).to_fs(:short) if d },
          temps: data['temps'].map { |t| t.to_f.round(1) if t }
        )
      else
        config.logger.warn('response was not OK!')
        nil
      end
    rescue => e
      config.logger.warn(e.detailed_message)
      nil
    end
  end
end
