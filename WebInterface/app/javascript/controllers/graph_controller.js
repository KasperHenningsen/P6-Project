import {Controller} from "@hotwired/stimulus"

export default class extends Controller {
    connect() {
        const datasets = JSON.parse(this.data.get("datasets"));
        const dates = JSON.parse(this.data.get("dates"));

        if (datasets && dates) {
            new Chart(document.getElementById('chart'), {
                type: "line",
                data: {
                    labels: dates,
                    datasets: datasets
                }
            });
        }
    }
}