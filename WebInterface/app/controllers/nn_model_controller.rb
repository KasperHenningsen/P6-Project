require 'date'
require 'async'

class NnModelController < ApplicationController
  def index
    Async do
      @data = Temperature.limit(1000).map { |t| [
        t.date,
        t.temp_min,
        t.temp_max]
      }
    end.wait
  end
end
