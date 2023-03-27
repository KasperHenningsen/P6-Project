require "test_helper"

class GraphsControllersControllerTest < ActionDispatch::IntegrationTest
  setup do
    @graphs_controller = graphs_controllers(:one)
  end

  test "should get index" do
    get graphs_controllers_url
    assert_response :success
  end

  test "should get new" do
    get new_graphs_controller_url
    assert_response :success
  end

  test "should create graphs_controller" do
    assert_difference("GraphsController.count") do
      post graphs_controllers_url, params: { graphs_controller: {  } }
    end

    assert_redirected_to graphs_controller_url(GraphsController.last)
  end

  test "should show graphs_controller" do
    get graphs_controller_url(@graphs_controller)
    assert_response :success
  end

  test "should get edit" do
    get edit_graphs_controller_url(@graphs_controller)
    assert_response :success
  end

  test "should update graphs_controller" do
    patch graphs_controller_url(@graphs_controller), params: { graphs_controller: {  } }
    assert_redirected_to graphs_controller_url(@graphs_controller)
  end

  test "should destroy graphs_controller" do
    assert_difference("GraphsController.count", -1) do
      delete graphs_controller_url(@graphs_controller)
    end

    assert_redirected_to graphs_controllers_url
  end
end
