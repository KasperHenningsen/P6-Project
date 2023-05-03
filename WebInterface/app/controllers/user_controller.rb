class UserController < ApplicationController
  before_action :get_user_id, only: [:show]

  def show
    @user = User.find(params[:id])
  end

  private

  def get_user_id
    params.require(:id)
  end
end
