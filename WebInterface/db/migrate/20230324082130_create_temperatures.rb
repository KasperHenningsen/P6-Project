class CreateTemperatures < ActiveRecord::Migration[7.0]
  def change
    create_table :temperatures do |t|
      t.datetime :date
      t.float :temp_min
      t.float :temp_max

      t.timestamps
    end
  end
end
