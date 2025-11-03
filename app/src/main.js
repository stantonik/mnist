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
import { cvtImageToMNISTInput, cvtMNISTInputToImage, softmax, indexArray, makeImageLink } from './utils';

let loopRunning = false;

// Prediction chart
let predictionsChart;
const rootStyles = getComputedStyle(document.documentElement);
const topBarColor = rootStyles.getPropertyValue('--color-primary').trim();
const barColor = rootStyles.getPropertyValue('--color-primary-hover').trim();
const textColor = rootStyles.getPropertyValue('--text-primary').trim();
const textSecColor = rootStyles.getPropertyValue('--text-secondary').trim();

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
                            color: textColor,
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: { color: textSecColor },
                        grid: { color: 'transparent' },
                        beginAtZero: true,
                        reverse: true
                    },
                    x: {
                        ticks: { color: textSecColor },
                        grid: { color: 'transparent' },
                        beginAtZero: true
                    }
                }
            }
        }
    );

    updatePredictions(p);
}

function updatePredictions(p) {
    if (!predictionsChart) return;

    // Sort descending by values
    const pSortedArray = Object.entries(p).sort((a, b) => b[1] - a[1]);

    // Separate keys and values for Chart.js
    const labels = pSortedArray.map(([key]) => key);
    const values = pSortedArray.map(([_, value]) => value);

    const maxValue = Math.max(...values);
    const backgroundColors = values.map(value => value === maxValue ? topBarColor : barColor);

    predictionsChart.data.labels = labels;
    predictionsChart.data.datasets[0].data = values;
    predictionsChart.data.datasets[0].backgroundColor = backgroundColors;
    predictionsChart.update();
}

// Entry point
setupCanvas();
await setupInference();

const p0 = Array.from({ length: 10 }, () => 0);
setupPredictions(indexArray(p0));

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

    setTimeout(loopInference, 500); // call again after 0.5s
}

// Event listeners
document.getElementById("run").addEventListener("click", (e) => {
    const btn = e.target;
    if (loopRunning) {
        btn.textContent = "Run Model";
        loopRunning = false;
    } else {
        btn.textContent = "Stop Model";
        loopRunning = true;
        loopInference();
    }
})
