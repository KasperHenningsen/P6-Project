class Dataset < ApplicationRecord
  has_many :data_point, dependent: :destroy
end
