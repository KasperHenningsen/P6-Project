class GraphController < ApplicationController
  def show
    setting = Setting.find(params[:id])
    dataset = Dataset.find(params[:dataset_id])

    @datasets = DataPoint.order(:date).where(identifier: "Actual", dataset_id: dataset.id)
    @dates = DataPoint.select(:date).order(:date).where(identifier: "Actual", dataset_id: dataset.id)

    setting.models.split(' ').each do |model|
      model_data = DataPoint.order(:date).where(identifier: model.strip, dataset_id: dataset.id)
      @datasets += model_data if model_data.any?
    end

    render 'graph/graph'
  end
end
