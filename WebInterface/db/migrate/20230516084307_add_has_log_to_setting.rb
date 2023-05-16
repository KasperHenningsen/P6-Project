class AddHasLogToSetting < ActiveRecord::Migration[7.0]
  def change
    add_column :settings, :has_log, :boolean, default: false
  end
end
