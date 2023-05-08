class GraphController < ApplicationController
  def show
    dataset = Dataset.find(params[:dataset_id])

    @datasets = DataPoint.select("identifier, json_group_array(date) AS dates, json_group_array(temp) AS temps")
                         .where(dataset_id: dataset.id)
                         .group(:identifier)
                         .order(:date)
                         .map { |d| {
                           identifier: d.identifier,
                           dates: JSON.parse(d.dates),
                           temps: JSON.parse(d.temps)
                         } }

    @dates = DataPoint.where(identifier: "Actual", dataset_id: dataset.id)
                      .order(:date)
                      .pluck(:date)
                      .map { |d| d.to_fs(:short) }
  end
end
