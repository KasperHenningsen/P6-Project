require 'sidekiq/web'

Rails.application.routes.draw do
  devise_for :users
  resources :settings
  
  # Require User account to access:
  authenticate :user do
    root 'pages#home'

    # Pages
    get 'loading', to: 'pages#spinner'
    get 'graph', to: 'pages#graph'

    # User
    get 'profile', to: 'user#show'

    # Setting
    delete 'setting', to: 'settings#delete'

    # Feature Matrix
    get 'featurematrix', to: 'feature_matrix#index'
  end

  # Only admin can access these sites:
  authenticate :user, lambda { |u| u.admin? } do
    mount Sidekiq::Web => '/sidekiq'
  end
end
