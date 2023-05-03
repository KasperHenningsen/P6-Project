class CreatePredictions < ActiveRecord::Migration[7.0]
  def change
    create_table :predictions do |t|
      t.belongs_to :setting, null: false, foreign_key: true
      t.float :temp
      t.datetime :date
    end
  end
end
