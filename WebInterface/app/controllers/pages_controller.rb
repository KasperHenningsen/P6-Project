class PagesController < ApplicationController
  def home
  end

  def options
    @data_options = get_options
  end

  private

  def get_options
    [["From API", "api"], ["Upload CSV", "csv"], ["Manual entry", "manual"]]
  end
end
