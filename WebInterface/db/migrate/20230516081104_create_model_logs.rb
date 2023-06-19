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

      t.string :target_column, null: true

      t.integer :sequence_length, null: true
      t.integer :target_length, null: true
      t.integer :batch_size, null: true
      t.integer :epochs, null: true
      t.integer :gradient_clipping, null: true
      t.integer :input_channels, null: true
      t.integer :input_size, null: true
      t.integer :hidden_size, null: true
      t.integer :output_size, null: true
      t.integer :depth, null: true
      t.integer :kernel_size, null: true
      t.integer :dilation_base, null: true
      t.integer :number_of_features, null: true
      t.integer :number_of_layers, null: true
      t.integer :convolution_channels, null: true
      t.integer :residual_channels, null: true
      t.integer :skip_channels, null: true
      t.integer :end_channels, null: true
      t.integer :tangent_alpha, null: true
      t.integer :d_model, null: true
      t.integer :nhead, null: true
      t.integer :dim_feedfoward, null: true

      t.float :learning_rate, null: true
      t.float :training_size, null: true
      t.float :dropout, null: true
      t.float :propagation_alpha, null: true

      t.boolean :use_output_convolution, null: true
      t.boolean :build_adjacency_matrix, null: true
    end
  end
end

