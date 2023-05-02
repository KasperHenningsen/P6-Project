Rails.application.routes.draw do
  resources :settings

  root 'pages#home'

  # Pages
  get 'graph', to: 'pages#graph'

  # Feature Matrix
  get 'featurematrix', to: 'feature_matrix#index'
end
