class AddUserToSetting < ActiveRecord::Migration[7.0]
  def change
    add_reference :users, :settings, null: true, foreign_key: true
  end
end
