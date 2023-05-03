# README

## Deployment

Local deployment of the rails application can be done by using the
command ```rails s``` in the ``/WebInterface`` directory.
Optionally the parameter ``-p [PORT]`` can be used to specify
the port and ``-e [ENVIRONMENT]`` can be used to specify the runtime environment.
Other arguments can be found [here](https://guides.rubyonrails.org/command_line.html#bin-rails-server)

## Devise

[Devise](https://github.com/heartcombo/devise) is a authentication solution for Rails.

## Sidekiq

[Sidekiq](https://github.com/sidekiq/sidekiq) is used as a job scheduler and can be deployed by running
the following command inside /WebInterface:

``bundle exec sidekiq -C config/sidekiq.yml``

The Sidekiq dashboard can be accessed by:

* Logging in as a admin user and pressing the "Monitor jobs" button, in the navbar dropdown.
* Accessing [``/sidekiq``](http://localhost:3000/sidekiq) and then logging in.
