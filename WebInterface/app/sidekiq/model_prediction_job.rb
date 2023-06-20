require "http"

class ModelPredictionJob
  include Sidekiq::Job

  def perform(setting_id)
    setting = Setting.find(setting_id)
    models = setting.models.split(' ')
    horizon = setting.horizon
    start_date = setting.start_date.iso8601
    end_date = setting.end_date.iso8601
    data_points = []

    models.each do |model|
      unless setting_exist(models, start_date, end_date, horizon)
        response = send_request(model.downcase, horizon, start_date, end_date)
        data_points << format_responses(response, model, setting.id)
      end
    end

    save_dataset(data_points)

    setting.has_dataset = true
    setting.save!
  end

  private

  def send_request(model, horizon, start_date, end_date)
    base_url = ENV["MODEL_API_URL"]
    req_url = URI.parse("#{base_url}/predictions/?model=#{model.downcase}&horizon=#{horizon}&start_date=#{start_date}&end_date=#{end_date}")
    HTTP.get(req_url)
  end

  def format_responses(response, model, dataset_id)
    data_points = []

    unless response.to_s.empty? || response.code != 200
      data = JSON.parse(response.to_s)
      data['dates'].each_with_index do |date, i|
        temp = data['temps'][i].round(1) if data['temps'][i]

        data_points << DataPoint.new({
                                       dataset_id: dataset_id,
                                       identifier: model,
                                       date: DateTime.parse(date),
                                       temp: temp
                                     })
      end
    end

    data_points
  end

  def save_dataset(data_points_arr)
    data_points = data_points_arr.flatten
    DataPoint.insert_all!(data_points.map(&:attributes))
  end

  def setting_exist(model, start_date, end_date, horizon)
    return Setting.where(start_date: start_date, end_date: end_date, horizon: horizon).where("models LIKE ?", "%#{model}%").exists?
  end
end
