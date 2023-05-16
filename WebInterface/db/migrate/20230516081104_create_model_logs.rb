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

      t.integer :seq_len
      t.integer :target_len
      t.string :target_col
      t.integer :batch_size
      t.integer :epochs
      t.float :learning_rate
      t.float :train_size

      t.integer :input_channels
      t.integer :hidden_size
      t.integer :kernel_size
      t.float :dropout, null: true

      t.timestamps
    end
  end
end
