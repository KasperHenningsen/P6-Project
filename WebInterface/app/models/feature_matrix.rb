class FeatureMatrix < ApplicationRecord
  def self.index
    require 'csv'

    path = Rails.root.join("app", "assets", "data", "correlation_coefficient_matrix.csv")
    csv = CSV.parse(File.read(path, :headers => true))

    require 'json'
    csv = csv.to_json
  end
end
