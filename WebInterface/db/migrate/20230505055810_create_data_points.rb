class CreateDataPoints < ActiveRecord::Migration[7.0]
  def change
    create_table :data_points do |t|
      t.string :identifier
      t.datetime :date
      t.float :temp
    end
  end
end
