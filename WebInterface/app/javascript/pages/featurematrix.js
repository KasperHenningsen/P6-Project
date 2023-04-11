import * as d3 from "d3-scale";

let features = ["temp_min", "temp_max", "date", "feels_like", "pressure"]
let feature_values = [[1, 1, 0.8, 0.6, 0.4], [1, 1, 0.8, 0.6, 0.3], [0.8, 0.8, 1, 0.5, 0.1], [0.6, 0.6, -1, 1, -0.2], [0.4, 0.3, 0.1, 0.2, 1]]

const cellColor = d3.scaleLinear().domain([1, -1]).range(["red", "blue"]);

const GRID = document.querySelector("#matrix")
GRID.style.gridTemplateColumns = `max-content repeat(${features.length + 1}, 30px)`

const ROWS = features.length;
const COLS = ROWS;
for (let i = 0; i <= ROWS; i++) {
    for (let j = 0; j <= COLS; j++) {
        const cell = document.createElement("div")
        cell.style.gridArea = `${i + 1} / ${j + 1}`

        let content = "";
        if (j <= i) {
            if (j === 0 && i < features.length) { // Left labels
                content = features[i];
                cell.classList.add("matrix-label-left")
            }
            else if (i === features.length) { // Bottom labels
                if (j !== 0) {
                    content = features[j - 1];
                    cell.classList.add("matrix-label-bottom")
                }
            }
            else { // Cells
                //content = feature_values[i][j - 1]; Correlation coefficient
                cell.classList.add("matrix-cell")
                cell.style.backgroundColor = cellColor(feature_values[i][j - 1])
            }
        }
        else {
            cell.classList.add("matrix-cell")
        }
        cell.innerText += content;
        GRID.appendChild(cell);
    }
}

document.querySelector("#legend").style.visibility = "visible";