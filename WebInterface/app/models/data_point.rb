class DataPoint < ApplicationRecord
  belongs_to :dataset, dependent: :destroy

  attr_accessor(:identifier, :date, :temp)
end
