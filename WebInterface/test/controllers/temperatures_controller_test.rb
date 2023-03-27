require "test_helper"

class TemperaturesControllerTest < ActionDispatch::IntegrationTest
  setup do
    @temperature = temperatures(:one)
  end

  test "should get index" do
    get temperatures_url
    assert_response :success
  end

  test "should get new" do
    get new_temperature_url
    assert_response :success
  end

  test "should create temperature" do
    assert_difference("Temperature.count") do
      post temperatures_url, params: { temperature: { date: @temperature.date, temp_max: @temperature.temp_max, temp_min: @temperature.temp_min } }
    end

    assert_redirected_to temperature_url(Temperature.last)
  end

  test "should show temperature" do
    get temperature_url(@temperature)
    assert_response :success
  end

  test "should get edit" do
    get edit_temperature_url(@temperature)
    assert_response :success
  end

  test "should update temperature" do
    patch temperature_url(@temperature), params: { temperature: { date: @temperature.date, temp_max: @temperature.temp_max, temp_min: @temperature.temp_min } }
    assert_redirected_to temperature_url(@temperature)
  end

  test "should destroy temperature" do
    assert_difference("Temperature.count", -1) do
      delete temperature_url(@temperature)
    end

    assert_redirected_to temperatures_url
  end
end
