Rails.application.routes.draw do
  root 'home#index'

  get 'models', to: 'nn_model#index'
  get 'about', to: 'about#index'
end
