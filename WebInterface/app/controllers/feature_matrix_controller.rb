class FeatureMatrixController < ApplicationController
  def index
    @features = FeatureMatrix.index
  end
end
