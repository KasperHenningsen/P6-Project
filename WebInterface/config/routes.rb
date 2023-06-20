require 'sidekiq/web'

Rails.application.routes.draw do
  devise_for :users
  resources :settings

  # Require User account to access:
  authenticate :user do
    root 'user#show'

    # Graph
    get 'graph', to: 'graph#show'

    # User
    get 'profile', to: 'user#show'

    # Setting
    delete 'setting', to: 'settings#delete'

    # Feature Matrix
    get 'featurematrix', to: 'feature_matrix#show'
  end

  # Only admin can access these sites:
  authenticate :user, lambda { |u| u.admin? } do
    mount Sidekiq::Web => '/sidekiq'
  end
end
