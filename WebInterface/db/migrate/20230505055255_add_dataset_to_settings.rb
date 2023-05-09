class AddDatasetToSettings < ActiveRecord::Migration[7.0]
  def change
    add_reference :settings, :datasets, null: true, foreign_key: true
  end
end
