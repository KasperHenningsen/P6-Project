require 'http'

class ActualValueJob
  include Sidekiq::Job

  def perform(dataset_id, start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}/actuals/subset?start_date=#{start_date}&end_date=#{end_date}"

    response = HTTP.get(url)

    unless response.to_s.nil? || response.code != 200
      data = JSON.parse(response.to_s)
      puts "ASSSS: #{data}"
      dates = data['dates'].map { |d| DateTime.parse(d) if d }
      temps = data['temps'].map { |t| t.to_f.round(2) if t }

      dates.each_with_index do |date, index|
        data_point = DataPoint.create!(dataset_id: dataset_id, identifier: "Actual", date: date, temp: temps[index])
        data_point.save!
      end
    end
  end
end