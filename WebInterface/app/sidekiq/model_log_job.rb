class ModelLogJob
  include Sidekiq::Job

  def perform(setting_id)
    setting = Setting.find(setting_id)
    models = setting.models.split(' ')
    horizon = setting.horizon

    model_logs = []

    models.each do |model|
      unless setting_exist(models, horizon)
        response = send_request(model.downcase, horizon)
        model_logs << format_response(response, horizon)
      end
    end

    ModelLog.insert_all(model_logs.map(&:attributes))
  end

  private

  def send_request(model, horizon)
    base_url = ENV["MODEL_API_URL"]
    req_url = URI.parse("#{base_url}/logs?model=#{model.downcase}&horizon=#{horizon}")
    return HTTP.get(req_url)
  end

  def format_response(response, horizon)
    new_log = nil
    unless response.to_s.empty? || response.code != 200
      json_log = JSON.parse(response.to_s)

      new_log = ModelLog.new({
                               model: json_log['model'],
                               horizon: horizon,
                               trained_at: json_log['trained_at'],

                               val_mae: json_log['loss']['validation']['mae'],
                               val_smape: json_log['loss']['validation']['smape'],
                               val_rmse: json_log['loss']['validation']['rmse'],

                               test_mae: json_log['loss']['test']['mae'],
                               test_smape: json_log['loss']['test']['smape'],
                               test_rmse: json_log['loss']['test']['rmse'],

                               target_column: json_log['model_parameters']['target_col'],

                               sequence_length: json_log['model_parameters']['seq_len'],
                               target_length: json_log['model_parameters']['target_len'],
                               batch_size: json_log['model_parameters']['batch_size'],
                               epochs: json_log['model_parameters']['epochs'],
                               gradient_clipping: json_log['model_parameters']['grad_clipping'],
                               input_channels: json_log['hyperparameters']['input_channels'],
                               input_size: json_log['hyperparameters']['input_size'],
                               hidden_size: json_log['hyperparameters']['hidden_size'],
                               output_size: json_log['hyperparameters']['output_size'],
                               depth: json_log['hyperparameters']['depth'],
                               kernel_size: json_log['hyperparameters']['kernel_size'],
                               dilation_base: json_log['hyperparameters']['dilation_base'],
                               number_of_features: json_log['hyperparameters']['num_features'],
                               number_of_layers: json_log['hyperparameters']['num_layers'],
                               convolution_channels: json_log['hyperparameters']['conv_channels'],
                               residual_channels: json_log['hyperparameters']['residual_channels'],
                               skip_channels: json_log['hyperparameters']['skip_channels'],
                               end_channels: json_log['hyperparameters']['end_channels'],
                               tangent_alpha: json_log['hyperparameters']['tan_alpha'],
                               d_model: json_log['hyperparameters']['d_model'],
                               nhead: json_log['hyperparameters']['nhead'],
                               dim_feedfoward: json_log['hyperparameters']['dim_feedfoward'],

                               learning_rate: json_log['model_parameters']['learning_rate'],
                               training_size: json_log['model_parameters']['train_size'],
                               dropout: json_log['hyperparameters']['dropout'],
                               propagation_alpha: json_log['hyperparameters']['prop_alpha'],

                               use_output_convolution: json_log['hyperparameters']['use_output_convolution'],
                               build_adjacency_matrix: json_log['hyperparameters']['build_adj_matrix']
                             })
    end

    return new_log
  end

  def save_logs(logs)
    ModelLog.insert_all(logs.map(&:attributes))
  end

  def setting_exist(model, horizon)
    return ModelLog.where(model: model, horizon: horizon).exists?
  end
end

