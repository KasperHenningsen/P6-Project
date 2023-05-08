class CreateDatasets < ActiveRecord::Migration[7.0]
  def change
    create_table :datasets do |t|
      t.belongs_to :setting

      t.timestamps
    end
  end
end
