class CreateGraphsControllers < ActiveRecord::Migration[7.0]
  def change
    create_table :graphs_controllers do |t|

      t.timestamps
    end
  end
end
