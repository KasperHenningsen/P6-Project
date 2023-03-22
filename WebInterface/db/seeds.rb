
'''
100.times do |i|
  NnModelController.create(title: "Title: #{i}",
                           date: Time.now - i.days,
                           temp: rand(0..100)
  )
end
'''