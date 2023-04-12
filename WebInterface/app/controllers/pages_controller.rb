class PagesController < ApplicationController
  def Home
  end

  def options
    @data_options = get_options
  end

  private

  def get_options
    return [["From API", "api"], ["Upload CSV", "csv"], ["Manual entry", "manual"]]
  end
end
