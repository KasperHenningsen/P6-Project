import { Controller } from "@hotwired/stimulus";
import * as d3 from "d3-scale";

// Connects to data-controller="heatmap--featurematrix"

export default class extends Controller {
    matrixData;
    features;
    prevSearch;

    connect() {
        this.initializeData();
        this.buildMatrix();
    }

    initializeData() {
        this.matrixData = JSON.parse(this.data.get("features"));
        this.features = this.matrixData[0];
        this.features[0] = "";
    }

    buildMatrix() {
        const cellColor = d3.scaleLinear().domain([1, -1]).range(["red", "blue"]);

        const GRID = document.querySelector("#matrix");
        GRID.style.gridTemplateColumns = `max-content repeat(${this.features.length}, minmax(12px, 2.5vh))`;

        const ROWS = this.features.length;
        for (let i = 1; i <= ROWS; i++) {
            for (let j = 0; j < ROWS; j++) {
                const cell = document.createElement("div");
                let id = i + "/" + (j + 1);

                cell.setAttribute("id", `${id}`)
                cell.style.gridArea = `${i} / ${j + 1}`;
                let content = "";

                if (i === ROWS) { //bottom label
                    if (j > 0) {
                        const label = document.createElement("div");
                        content = this.features[j];
                        label.innerText += content;

                        label.classList.add("label", "bottom");
                        cell.classList.add("container-bottom");
                        cell.appendChild(label);
                    } else {
                        cell.classList.add("matrix-cell") // Corner cell
                    }
                } else if (j === 0) { //left label
                    content = this.matrixData[i][j];
                    cell.classList.add("label");
                    cell.innerText += content;
                }
                else if (j < i) { //data cells
                    cell.style.backgroundColor = cellColor(this.matrixData[i][j]);
                    cell.classList.add("matrix-cell", "data-cell");

                    let controller = "heatmap--featurematrix"
                    cell.setAttribute("data-action", `mouseover->${controller}#addHoverBorder mouseout->${controller}#removeHoverBorder`);

                    let content = parseFloat(this.matrixData[i][j]).toPrecision(3);
                    const tooltip = document.createElement("div");
                    tooltip.classList.add("tooltiptext");
                    tooltip.innerText += `${this.features[j] + " / "  + this.matrixData[i][0] + " \n " + content}`
                    cell.appendChild(tooltip)
                }
                else {
                    cell.classList.add("matrix-cell");
                }
                GRID.appendChild(cell);
            }
        }
        if (this.matrixData.length !== 0) document.querySelector("#legend").style.visibility = "visible";
    }

    searchMatrix(event) {
        if (event.currentTarget.value) {
            if (event.currentTarget.value < this.prevSearch) {
                this.clearLowlights()
            }
            
            this.prevSearch = event.currentTarget.value;
            const ROWS = this.features.length;
            for (let i = 1; i < ROWS; i++) {
                for (let j = 0; j < i; j++) {
                    if (Math.abs(Number(this.matrixData[i][j])) < event.currentTarget.value) {
                        let id = i + "/" + (j + 1);
                        let TARGET = document.querySelector("#" + CSS.escape(id));
                        if (TARGET) TARGET.classList.add("lowlight");
                    }
                }
            }
        }
    }

    clearLowlights() {
        const TARGETS = document.querySelectorAll(".lowlight");
        TARGETS.forEach(elem => elem.classList.remove("lowlight"));
    }

    addHoverBorder(event) {
        let gridPos = event.currentTarget.id.split('/');
        let row = gridPos[0];
        let col = gridPos[1];

        for (let i = row; i <= this.features.length; i++) {
            let cell = `${i}/${gridPos[1]}`
            document.querySelector("#" + CSS.escape(cell)).classList.add("border-ver")
        }

        for (let i = col; i > 0; i--) {
            let cell = `${gridPos[0]}/${i}`
            document.querySelector("#" + CSS.escape(cell)).classList.add("border-hor")
        }
    }

    removeHoverBorder() {
        let borderCells = document.querySelectorAll(".border-ver, .border-hor")
        borderCells.forEach(cell => cell.classList.remove("border-ver", "border-hor"))
    }
}