Rails.application.routes.draw do
  resources :temperatures
  resources :graphs_controllers

  root 'pages#home'

  get 'models', to: 'nn_model#index'
  get 'about', to: 'pages#about'
end
