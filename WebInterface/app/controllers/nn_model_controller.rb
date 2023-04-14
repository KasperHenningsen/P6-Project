class NnModelController < ApplicationController

  def index
    $data = params["_json"].to_s
  end

  def json
  end
end
