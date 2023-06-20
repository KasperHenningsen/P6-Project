class AddHasDatasetToSetting < ActiveRecord::Migration[7.0]
  def change
    add_column :settings, :has_dataset, :boolean
  end
end
