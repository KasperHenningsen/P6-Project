import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  connect() {
    let [date_min, date_max] = this.initializeData();
    
    flatpickr(".flatpickr-date-field", {
      allowInput: true,
      enableTime: true,
      time_24hr: true,
      hourIncrement: 1,
      minuteIncrement: 60,
      defaultMinute: 0,
      enableSeconds: false,
      minDate: date_min,
      maxDate: date_max,
      dateFormat: "d-m-Y H:i",
    });
  }

  initializeData() {
    return this.data.get("dates").split(";");
  }
}
