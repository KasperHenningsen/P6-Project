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
        model_logs << format_responses(response)
      end
    end

    save_logs(model_logs)

    setting.has_log = true
    setting.save!
  end

  private

  def send_request(model, horizon)
    base_url = ENV["MODEL_API_URL"]
    req_url = URI.parse("#{base_url}/predictions/?model=#{model.downcase}&horizon=#{horizon}") # Add "log" somewhere...
    return HTTP.get(req_url)
  end

  def format_responses(response)
    new_log = nil
    unless response.to_s.nil? || response.code != 200
      json_log = JSON.parse(response.to_s)

      new_log = ModelLog.new(
        model: json_log['model'],
        trained_at: json_log['trained_at'],
        val_mae: json_log['validation']['mae'],
        val_smape: json_log['validation']['smape'],
        val_rmse: json_log['validation']['rmse'],
        test_mae: json_log['test']['mae'],
        test_smape: json_log['test']['smape'],
        test_rmse: json_log['test']['rmse'],
        seq_len: json_log['hyperparameters']['seq_len'],
        target_len: json_log['hyperparameters']['target_len'],
        target_col: json_log['hyperparameters']['target_col'],
        batch_size: json_log['hyperparameters']['batch_size'],
        epochs: json_log['hyperparameters']['epochs'],
        learning_rate: json_log['hyperparameters']['learning_rate'],
        train_size: json_log['hyperparameters']['train_size'],
        input_channels: json_log['model_parameters']['input_channels'],
        hidden_size: json_log['model_parameters']['hidden_size'],
        kernel_size: json_log['model_parameters']['kernel_size'],
        dropout: json_log['model_parameters']['dropout']
      )
    end

    return new_log
  end

  def save_logs(logs)
    log_attrs = logs.map { |log| log.attributes }
    ModelLog.insert_all(log_attrs)
  end

  def setting_exist(model, horizon)
    return ModelLog.where(horizon: horizon).where("models LIKE ?", "%#{model}%").exists?
  end
end
