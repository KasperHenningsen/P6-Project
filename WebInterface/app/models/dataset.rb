class Dataset < ApplicationRecord
  belongs_to :setting
  has_many :data_points, dependent: :destroy
end
