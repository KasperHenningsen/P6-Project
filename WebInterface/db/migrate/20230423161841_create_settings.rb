class CreateSettings < ActiveRecord::Migration[7.0]
  def change
    create_table :settings do |t|
      t.datetime :start_date, null: false
      t.datetime :end_date, null: false
      t.integer :horizon, null: false
      t.string :models, null: false

      t.timestamps
    end
  end
end
