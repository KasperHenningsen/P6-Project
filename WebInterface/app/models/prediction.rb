class Prediction < ApplicationRecord
  belongs_to :setting, dependent: :destroy
end
