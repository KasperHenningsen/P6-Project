class RemoveIdAndTimestampsFromTemperatures < ActiveRecord::Migration[6.1]
  def change
    remove_column :temperatures, :id, :bigint
    remove_column :temperatures, :created_at, :datetime
    remove_column :temperatures, :updated_at, :datetime
  end
end
