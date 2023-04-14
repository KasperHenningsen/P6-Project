import { Controller } from "@hotwired/stimulus"

// Connects to data-controller="pages--home"

export default class extends Controller {
  connect() {
    //TODO: testing remove later
    const start_date = document.getElementById("start-date-input");
    const end_date = document.getElementById("end-date-input");

    start_date.value = new Date("2020-01-01T01:00").toISOString().slice(0, 16);
    end_date.value = new Date("2020-01-01T03:00").toISOString().slice(0, 16);
  }

  initHome() {
    const checkboxes = document.querySelectorAll('input[type=checkbox]');
    const dataSelector = document.querySelector('#data_option');
    const confirmButton = document.querySelector('#submit-button');


    function enableApplyButton() {
      let enable = false;
      checkboxes.forEach(checkbox => {
        if (checkbox.checked) {
          enable = true;
        }
      });
      if (dataSelector.value && enable) {
        confirmButton.disabled = false;
      } else {
        confirmButton.disabled = true;
      }
    }

    checkboxes.forEach(checkbox => {
      checkbox.addEventListener('change', enableApplyButton);
    });

    dataSelector.addEventListener('change', enableApplyButton);
  }
}
