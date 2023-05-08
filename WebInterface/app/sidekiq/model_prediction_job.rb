class ModelPredictionJob
  include Sidekiq::Job

  def perform(setting_id, dataset_id)
    datasets = []
    setting = Setting.find(setting_id)
    horizon = setting.horizon
    start_date = setting.start_date.iso8601
    end_date = setting.end_date.iso8601
    response = []

    setting.models.split(' ').each do |model|
      response << send_request(model.downcase, horizon, start_date, end_date)
      datasets << format_response(response, model.downcase)
      if save_datasets(dataset_id, datasets)
        setting.has_dataset = true
      end
    end
  end

  private

  def send_request(model, horizon, start_date, end_date)
    base_url = ENV["MODEL_API_URL"]
    req_url = URI.parse("#{base_url}/predictions/models/#{model.downcase}?horizon=#{horizon}&start_date=#{start_date}&end_date=#{end_date}")
    return Net::HTTP.get_response(req_url)
  end

  def format_response(response, model)
    if response.code == '200'
      data = JSON.parse(response.body)
      data_points = []

      data['dates'].each_with_index do |date, i|
        temp = data['temps'][i].to_f.round(1) if data['temps'][i]

        data_points << DataPoint.new(
          identifier: model,
          date: DateTime.parse(date).to_fs(:short),
          temp: temp
        )
      end

      return data_points
    end
  end

  def save_datasets(dataset_id, datasets)
    datasets.each do |data_point_collection|

      data_point_collection.each do |datapoint|
        datapoint.dataset_id = dataset_id
        datapoint.save!
      end
    end

    return true
  end
end
