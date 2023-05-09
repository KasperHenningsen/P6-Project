require 'json'
require 'uri'
require 'net/http'
require 'csv'

class FeatureMatrixController < ApplicationController
  def index
    @features = getData
  end

  def getData
    uri = URI("#{ENV['MODEL_API_URL']}/featurematrix")
    res = Net::HTTP.get_response(uri)
    data = res.body if res.is_a?(Net::HTTPSuccess)
    CSV.parse(data).to_json
  end
end
