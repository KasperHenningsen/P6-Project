class ActualValueJob
  include Sidekiq::Job
  sidekiq_options dead: false

  def perform(dataset_id, start_date, end_date)
    base_url = ENV['MODEL_API_URL']
    url = "#{base_url}/dataset/actuals?start_date=#{start_date}&end_date=#{end_date}"

    response = HTTParty.get(url)
    if response.code == '200'
      data = JSON.parse(response.body)

      dates = data['dates'].map { |d| DateTime.parse(d).to_fs(:short) if d }
      temps = data['temps'].map { |t| t.to_f.round(2) if t }

      dates.each_with_index do |index, date|
        data_point = DataPoint.create!(dataset_id: dataset_id, identifier: "Actual", date: date, temp: temps[index])
        data_point.save!
      end
    end
  end
end