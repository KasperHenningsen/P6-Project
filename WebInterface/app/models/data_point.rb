class DataPoint < ApplicationRecord
  belongs_to :dataset, dependent: :destroy
end
