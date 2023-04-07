class PagesController < ApplicationController
  def Home
    @model_list = get_models
    @data_options = get_data_options
  end

  def mlp
  end

  def gru
  end

  def rnn
  end

  def lstm
  end

  def mtgnn
  end

  private
  def get_models
    return (["GRU", "LSTM", "MLP", "MTGNN", "RNN"]).sort
  end

  def get_data_options
    return [["Manual", "manual"], ["From API", "api"], ["Upload dataset", "upload"]]
  end
end
