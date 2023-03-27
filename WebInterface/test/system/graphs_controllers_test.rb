require "application_system_test_case"

class GraphsControllersTest < ApplicationSystemTestCase
  setup do
    @graphs_controller = graphs_controllers(:one)
  end

  test "visiting the index" do
    visit graphs_controllers_url
    assert_selector "h1", text: "Graphs controllers"
  end

  test "should create graphs controller" do
    visit graphs_controllers_url
    click_on "New graphs controller"

    click_on "Create Graphs controller"

    assert_text "Graphs controller was successfully created"
    click_on "Back"
  end

  test "should update Graphs controller" do
    visit graphs_controller_url(@graphs_controller)
    click_on "Edit this graphs controller", match: :first

    click_on "Update Graphs controller"

    assert_text "Graphs controller was successfully updated"
    click_on "Back"
  end

  test "should destroy Graphs controller" do
    visit graphs_controller_url(@graphs_controller)
    click_on "Destroy this graphs controller", match: :first

    assert_text "Graphs controller was successfully destroyed"
  end
end
