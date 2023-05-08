require "http"

class ModelPredictionJob
  include Sidekiq::Job

  def perform(setting_id, dataset_id)
    setting = Setting.find(setting_id)
    models = setting.models.split(' ')
    horizon = setting.horizon
    start_date = setting.start_date.iso8601
    end_date = setting.end_date.iso8601

    models.each do |model|
      unless setting_exist(models, start_date, end_date, horizon)
        response = send_request(model.downcase, horizon, start_date, end_date)
        format_responses(response, model, dataset_id)
      end
    end

    setting.has_dataset = true
    setting.save!
  end

  private

  def send_request(model, horizon, start_date, end_date)
    base_url = ENV["MODEL_API_URL"]
    req_url = URI.parse("#{base_url}/predictions/?model=#{model.downcase}&horizon=#{horizon}&start_date=#{start_date}&end_date=#{end_date}")
    return HTTP.get(req_url)
  end

  def format_responses(response, model, dataset_id)
    unless response.to_s.nil? || response.code != 200
      data = JSON.parse(response.to_s)
      data['dates'].each_with_index do |date, i|
        temp = data['temps'][i].round(1) if data['temps'][i]

        data_point = DataPoint.create(
          dataset_id: dataset_id,
          identifier: model,
          date: DateTime.parse(date),
          temp: temp
        )

        data_point.save!
      end
    end
  end

  def setting_exist(model, start_date, end_date, horizon)
    return Setting.where(start_date: start_date, end_date: end_date, horizon: horizon).where("models LIKE ?", "%#{model}%").exists?
  end
end
