class CreateDatasets < ActiveRecord::Migration[7.0]
  def change
    create_table :datasets do |t|

      t.timestamps
    end
  end
end
