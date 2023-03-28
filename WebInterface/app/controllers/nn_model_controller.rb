require 'date'

class NnModelController < ApplicationController
  def index
    @data = Temperature.limit(1000).offset(0).map { |t| [
      t.date * 1000,
      t.temp_min,
      t.temp_max
    ]}
  end
end
