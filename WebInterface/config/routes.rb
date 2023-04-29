Rails.application.routes.draw do
  resources :settings

  root 'pages#home'

  # Pages
  get 'load', to: 'pages#load'

  # Feature Matrix
  get 'featurematrix', to: 'feature_matrix#index'
end
