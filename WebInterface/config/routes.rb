Rails.application.routes.draw do
  get 'graph/index'
  resources :settings

  root 'pages#home'

  # Pages
  get 'graph', to: 'graph#show'
  resources :settings

  # Feature Matrix
  get 'featurematrix', to: 'feature_matrix#index'
end
