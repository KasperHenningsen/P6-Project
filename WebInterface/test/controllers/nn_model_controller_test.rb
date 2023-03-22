require "test_helper"

class NnModelControllerTest < ActionDispatch::IntegrationTest
  test "should get index" do
    get nn_model_index_url
    assert_response :success
  end
end
