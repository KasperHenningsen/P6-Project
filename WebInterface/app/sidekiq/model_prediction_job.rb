class ModelPredictionJob
  include Sidekiq::Job

  def perform(model, horizon, start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}#{model}?horizon=#{horizon}&start_date=#{start_date}&end_date=#{end_date}"

    begin
      response = HTTParty.get(url)
      response.body
    rescue => e
      return nil
    end
  end
end
