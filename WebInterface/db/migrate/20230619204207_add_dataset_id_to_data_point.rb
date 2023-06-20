class AddDatasetIdToDataPoint < ActiveRecord::Migration[7.0]
  def change
    add_column :data_points, :dataset_id, :integer
  end
end
