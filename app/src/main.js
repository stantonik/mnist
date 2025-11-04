/*
 * main.js
 * Copyright (C) 2025 stantonik <stantonik@stantonik-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

import './style.css';
import Chart from 'chart.js/auto'
import { getCanvasImageData, setupCanvas } from './drawing';
import { setupInference, makePrediction } from './inference';
import { cvtImageToMNISTInput, cvtMNISTInputToImage, softmax, indexArray, makeImageLink, PALETTE } from './utils';

let loopRunning = false;
let predictionsChart;

function setupPredictions(p) {
    predictionsChart = new Chart(
        document.getElementById('predictions'),
        {
            type: 'bar',
            data: {
                datasets: [
                    {
                        label: 'Predictions',
                    }
                ]
            },
            options: {
                indexAxis: 'x',
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        align: 'center',
                        labels: {
                            color: PALETTE.text.secondary,
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        display: false,
                        grid: { color: 'transparent' },
                        beginAtZero: true,
                        reverse: true
                    },
                    x: {
                        ticks: { color: PALETTE.text.secondary },
                        grid: { color: 'transparent' },
                        beginAtZero: true
                    }
                }
            }
        }
    );

    updatePredictions(p);
}

function updatePredictions(p = null, asc = false) {
    if (!predictionsChart) return;

    if (!p) {
        p = indexArray(Array.from({ length: 10 }, () => 0));
    }

    // Sort descending by values
    const pSortedArray = asc ? Object.entries(p).sort((a, b) => b[1] - a[1]) : Object.entries(p);

    // Separate keys and values for Chart.js
    const labels = pSortedArray.map(([key]) => key);
    const values = pSortedArray.map(([_, value]) => value);

    // Find the biggest value
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    const maxIndex = values.indexOf(maxValue);

    const backgroundColors = values.map(value => value === maxValue ? PALETTE.color.primary : PALETTE.color.primaryHover);

    predictionsChart.data.labels = labels;
    predictionsChart.data.datasets[0].data = values;
    predictionsChart.data.datasets[0].backgroundColor = backgroundColors;
    if (maxValue != minValue) {
        predictionsChart.options.scales.x.ticks.font = (ctx) => (ctx.tick.label == maxIndex ? { weight: 'bold' } : { weight: 'normal' });
        predictionsChart.options.scales.x.ticks.color = (ctx) => (ctx.tick.label == maxIndex ? PALETTE.color.primary : PALETTE.text.secondary);
    } else {
        predictionsChart.options.scales.x.ticks.font = { weight: 'normal' };
        predictionsChart.options.scales.x.ticks.color = PALETTE.text.secondary;
    }
    predictionsChart.update();
}

// Entry point
setupCanvas();
await setupInference();

setupPredictions();

async function loopInference() {
    if (!loopRunning) return;

    const imageData = getCanvasImageData(28, 28);
    const input = cvtImageToMNISTInput(imageData);

    // Download to preview input
    const link = document.getElementById("check-input")
    link.href = makeImageLink(cvtMNISTInputToImage(input, 28, 28), 28, 28);
    link.download = "input";

    const logits = await makePrediction(input);
    const p = softmax(logits);
    const pIdx = indexArray(p);
    console.log(p);
    console.log(pIdx);
    updatePredictions(pIdx);

    setTimeout(loopInference, 200); // call again after 0.2s
}

// Event listeners
document.getElementById("run").addEventListener("click", (e) => {
    const btn = e.target;
    if (loopRunning) {
        btn.textContent = "Run Model";
        loopRunning = false;
        updatePredictions();
    } else {
        btn.textContent = "Stop Model";
        loopRunning = true;
        loopInference();
    }
})
