class ModelPredictionJob
  include Sidekiq::Job
  sidekiq_options dead: false

  def perform(model, horizon, start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}#{model}?horizon=#{horizon}&start_date=#{start_date}&end_date=#{end_date}"

    response = HTTParty.get(url)
    response.body
  end
end
