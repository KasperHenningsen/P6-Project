class CreateModelLogs < ActiveRecord::Migration[7.0]
  def change
    create_table :model_logs do |t|
      t.string :model
      t.string :horizon
      t.datetime :trained_at

      t.float :val_mae
      t.float :val_smape
      t.float :val_rmse

      t.float :test_mae
      t.float :test_smape
      t.float :test_rmse

      t.integer :seq_len, null: true
      t.integer :target_len, null: true
      t.string :target_col, null: true
      t.integer :batch_size, null: true
      t.integer :epochs, null: true
      t.float :learning_rate, null: true
      t.float :train_size, null: true

      t.integer :input_channels, null: true
      t.integer :hidden_size, null: true
      t.integer :kernel_size, null: true
      t.integer :num_layers, null: true
      t.float :dropout, null: true
    end
  end
end

