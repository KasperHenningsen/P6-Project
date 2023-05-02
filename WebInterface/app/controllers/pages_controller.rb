require 'async'
require 'async/http/internet'

class PagesController < ApplicationController
  def home
  end

  def graph
    setting = Setting.find(params[:id])
    client = Async::HTTP::Internet.new
    tasks = get_model_data_async(setting)

    render 'pages/graph'
  end

  private

  Async def get_model_data_async(setting)
    base_url = ENV['MODEL_API_URL']
    models = setting.models.split(",")
    start_date = setting.start_date.strftime('%Y-%m-%dT%H:%M:%S.%L%z')
    end_date = setting.end_date.strftime('%Y-%m-%dT%H:%M:%S.%L%z')
    tasks = []

    Async do
      client = Async::HTTP::Internet.new

      models.each do |model|
        url = URI("#{base_url}#{model}?horizon=#{setting.horizon}&start_date=#{start_date}&end_date=#{end_date}")
        task = client.async.get(url).then(&:read)
        tasks << task
      end

      client&.close
    end

    return tasks
  end

  def some
    responses = []
    Async::Task::WaitAll(*tasks)

    tasks.each do |task|
      response = task.result
      if response.status == 200
        responses << response.read
      else
        return nil
      end
    end

    client.close
  end

  def some_other
    client = Async::HTTP::Internet.new
  end
end
