class ActualValuesJob
  include Sidekiq::Job
  sidekiq_options dead: false

  def perform(start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}/dataset/actuals?start_date=#{start_date}&end_date=#{end_date}"

    response = HTTParty.get(url)
    if response.code == '200'
      data = JSON.parse(response.body)
      return Dataset.new(
        identifier: "Actual",
        dates: data['dates'].map { |d| DateTime.parse(d).to_fs(:short) if d },
        temps: data['temps'].map { |t| t.to_f.round(2) if t }
      )
    else
      Rails.configuration.logger.warn("retrieved non-OK status from API: #{response.code}")
    end
  end
end
