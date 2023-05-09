# This file should contain all the record creation needed to seed the database with its default values.
# The data can then be loaded with the bin/rails db:seed command (or created alongside the database with db:setup).
#
# Examples:
#
#   movies = Movie.create([{ name: "Star Wars" }, { name: "Lord of the Rings" }])
#   Character.create(name: "Luke", movie: movies.first)

User.create!(email: "admin@example.com",
             password: "admin1234",
             password_confirmation: "admin1234",
             admin: true,
             username: "admin"
)

User.create!(email: "user@example.com",
             password: "user1234",
             password_confirmation: "user1234",
             admin: false,
             username: "user"
)