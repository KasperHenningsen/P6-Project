Rails.application.routes.draw do
  root 'pages#home'

  # DataInput
  get 'data', to: 'data_input#index'
  get 'data/upload', to: 'data_input#upload_data'
  get 'data/manual', to: 'data_input#manual_data'
  get 'data/api', to: 'data_input#api_data'

  # Pages
  get 'mlp', to: 'pages#mlp'
  get 'rnn', to: 'pages#rnn'
  get 'gru', to: 'pages#gru'
  get 'lstm', to: 'pages#lstm'
  get 'mtgnn', to: 'pages#mtgnn'

  # Feature Matrix
  get 'featurematrix', to: 'feature_matrix#index'
end
