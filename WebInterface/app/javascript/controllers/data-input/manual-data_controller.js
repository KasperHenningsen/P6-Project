import { Controller } from "@hotwired/stimulus"

// Connects to data-controller="data-input--manual-data"

export default class extends Controller {

  connect() {

  }

  async sendData(event) {
    const csrfToken = document.querySelector('meta[name="csrf-token"]').content;

    let data = "";
    let index = 0;

    while (document.getElementById(`row-${index}`) !== null) {
      const date = document.getElementById(`date-${index}`).innerHTML
      const temp = document.getElementById(`temp-input-${index}`).value
      const dew_point = document.getElementById(`dew-point-input-${index}`).value
      const pressure = document.getElementById(`pressure-input-${index}`).value
      const humidity = document.getElementById(`humidity-input-${index}`).value
      const rain = document.getElementById(`rain-one-hour-input-${index}`).value
      const snow = document.getElementById(`snow-one-hour-input-${index}`).value

      data += `[${date}, ${temp}, ${dew_point}, ${pressure}, ${humidity}, ${rain}, ${snow}],`;

      index++;
    }

    let json_data = JSON.stringify(data);

    await fetch('/models', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken
      },
      body: json_data
    });
  }
}
