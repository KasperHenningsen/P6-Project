require 'date'

class NnModelController < ApplicationController
  def index
    @data = Temperature.limit(1000).map { |t| [
      t.date,
      t.temp_min,
      t.temp_max]
    }
  end
end
