require 'http'

class ActualValueJob
  include Sidekiq::Job

  def perform(dataset_id, start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}/actuals/subset?start_date=#{start_date}&end_date=#{end_date}"

    response = HTTP.get(url)

    unless response.to_s.nil? || response.code != 200
      data = JSON.parse(response.to_s)

      dates = data['dates'].map { |d| DateTime.parse(d) if d }
      temps = data['temps'].map { |t| t.to_f.round(2) if t }

      data_points = dates.each_with_index.map do |date, index|
        { dataset_id: dataset_id, identifier: "Actual", date: date, temp: temps[index] }
      end

      DataPoint.insert_all!(data_points)
    end
  end
end