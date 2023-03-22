module HomeHelper

  # Random data for the graph
  def plot
    arr = []
    10.times do |i|
      arr << [i, rand(0..100)]
    end

    return arr
  end

end
