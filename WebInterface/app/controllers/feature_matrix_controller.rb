class FeatureMatrixController < ApplicationController
  def index
    @features = getData
  end

  def getData
    require 'json'
    require 'uri'
    require 'net/http'
    
    uri = URI('http://localhost:5000/featurematrix')
    res = Net::HTTP.get_response(uri)
    data = res.body if res.is_a?(Net::HTTPSuccess)
    CSV.parse(data).to_json
  end
end
