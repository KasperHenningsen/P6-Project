Rails.application.routes.draw do
  root 'pages#Home'

  get 'options', to: 'pages#options'
  get 'data', to: 'data_input#index'
  get 'models/json', to: 'nn_model#json'

  post 'data', to: 'data_input#upload'
  post 'models', to: 'nn_model#index'

  get 'data/csv', to: 'data_input#csv'
  get 'data/manual', to: 'data_input#manual'
  get 'data/api', to: 'data_input#api'

  # Pages
  get 'mlp', to: 'pages#mlp'
  get 'rnn', to: 'pages#rnn'
  get 'gru', to: 'pages#gru'
  get 'lstm', to: 'pages#lstm'
  get 'mtgnn', to: 'pages#mtgnn'
end
