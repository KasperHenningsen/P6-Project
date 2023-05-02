class GraphController < ApplicationController
  def show
    setting = Setting.find(params[:id])
    @labels = %w[one two three]
    @data = get_actuals(setting.start_date, setting.end_date)
    render('pages/graph')
  end

  private
  require('net/http')

  def get_actuals(start_date, end_date)
    start_date_iso = start_date.iso8601
    end_date_iso = end_date.iso8601
    req_url = URI("http://127.0.0.1:5000/predictions/gru?horizon=12&start_date=#{start_date_iso}&end_date=#{end_date_iso}")

    begin
      res = Net::HTTP.get_response(req_url)

      if res.code == 200
        res.body
      else
        config.logger.warn('response was not OK!')
        []
      end
    rescue
      config.logger.warn("ass")
      []
    end
  end
end
