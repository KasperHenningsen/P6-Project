# README

Something about the rails application....

* [Deployment](#deployment)
* [Device](#devise)
* [Sidekiq](#sidekiq)
* [Redis](#redis)

## Deployment

Local deployment of the rails application can be done by using the
command ``rails s`` in the ``/WebInterface`` directory.
Optionally the parameter ``-p [PORT]`` can be used to specify
the port and ``-e [ENVIRONMENT]`` can be used to specify the runtime environment.
Other arguments can be found [here](https://guides.rubyonrails.org/command_line.html#bin-rails-server)

## Devise

[Devise](https://github.com/heartcombo/devise) is a authentication solution for Rails.

For local development admin and user accounts can be generated using the seed.rb file.
An example of this can be seen below:

````
User.create!(email: "admin@admin.com",
             password: "admin1234",
             password_confirmation: "admin1234",
             admin: true,
             username: "admin"
) 

User.create!(email: "user@user.com",
             password: "user1234",
             password_confirmation: "user1234",
             admin: false,
             username: "user"
)
````

## Sidekiq

[Sidekiq](https://github.com/sidekiq/sidekiq) is used as a job scheduler and can be deployed by running
the following command inside /WebInterface while having a [Redis](#redis) server running:

````
bundle exec sidekiq -C config/sidekiq.yml
````

The Sidekiq dashboard can be accessed by:

* Logging in as a admin user and pressing the "Monitor jobs" button, in the navbar dropdown.
* Accessing [``/sidekiq``](http://localhost:3000/sidekiq) and then logging in.

## Redis

[Redis](https://redis.io/) is an open-source, in-memory data structure store that
can be used as a database, cache, and message broker. It supports a wide
range of data structures and provides high performance and scalability.
Redis can be deployd using this line, once installed:

````
service redis-server start
````
