class AddUserToSetting < ActiveRecord::Migration[7.0]
  def change
    add_reference :settings, :user, null: true, foreign_key: false
  end
end
