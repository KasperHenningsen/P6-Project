class HorizonListJob
  include Sidekiq::Job

  def perform
    [12, 24, 48]
  end

  def get_horizons
    base_url = ENV["MODEL_API_URL"]
    req_url = URI("#{base_url}/horizons")
    response = make_request(req_url)

    response.nil? ? nil : JSON.parse(response.body)
  end
end
