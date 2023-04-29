class PagesController < ApplicationController
  def home
    redirect_to new_setting_path
  end

  def load
  end
end
