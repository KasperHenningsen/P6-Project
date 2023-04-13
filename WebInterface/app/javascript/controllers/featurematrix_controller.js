import { Controller } from "@hotwired/stimulus";
import * as d3 from "d3-scale";

export default class extends Controller {
    connect() {
        this.buildMatrix()
    }

    buildMatrix() {
        let matrixData = JSON.parse(this.data.get("features"));

        let features = matrixData[0];
        features[0] = "";

        const cellColor = d3.scaleLinear().domain([1, -1]).range(["red", "blue"]);

        console.log(matrixData)

        const GRID = document.querySelector("#matrix");
        GRID.style.gridTemplateColumns = `max-content repeat(${features.length}, minmax(17px, 0.4fr))`;

        const ROWS = features.length;
        for (let i = 1; i <= ROWS; i++) {
            for (let j = 0; j < ROWS; j++) {
                const cell = document.createElement("div");
                cell.style.gridArea = `${i + 1} / ${j + 1}`;

                let content = "";
                if (i === ROWS) { //bottom label
                    if (j > 0) {
                        content = features[j];
                        cell.classList.add("matrix-label", "matrix-label-bottom");
                    } else cell.classList.add("matrix-cell")
                } else if (j === 0) { //left label
                    content = matrixData[i][j];
                    cell.classList.add("matrix-label");
                } else { //data cells
                    if (j < i) {
                        //content = parseFloat(matrixData[i][j]).toFixed(3); //Correlation coefficient
                        cell.style.backgroundColor = cellColor(matrixData[i][j]);
                    }
                    cell.classList.add("matrix-cell");
                }
                cell.innerText += content;
                GRID.appendChild(cell);
            }
        }
        if (matrixData.length !== 0) document.querySelector("#legend").style.visibility = "visible";
    }
}