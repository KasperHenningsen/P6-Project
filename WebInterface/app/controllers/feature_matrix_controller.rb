require 'json'
require 'uri'
require 'net/http'
require 'csv'

class FeatureMatrixController < ApplicationController
  def show
    begin
      if params[:dataset].present?
        @data_options = get_data_options
        @features = get_feature_matrix(params[:dataset])
      else
        @data_options = get_data_options
        @features = get_feature_matrix('Weather')
      end
    rescue => e
      flash[:error] = "Model API did not respond."
      redirect_to profile_path
    end
  end

  private

  def get_feature_matrix(dataset)
    uri = URI("#{ENV['MODEL_API_URL']}/featurematrix?dataset=#{dataset}")
    res = Net::HTTP.get_response(uri)
    data = res.body if res.is_a?(Net::HTTPSuccess)
    CSV.parse(data).to_json
  end

  def get_data_options
    %w[Weather Energy]
  end
end
