/*
 * drawing.js
 * Copyright (C) 2025 stantonik <stantonik@stantonik-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

import penSvg from './assets/pen_tool.svg'
import eraserSvg from './assets/eraser_tool.svg'
import clearSvg from './assets/clear_tool.svg'

// Parameters
const radiusMax = 100;
const radiusMin = 5;
const radiusStep = 1;
let radiusCurrent = 10;
const canvasWidth = 512;
const canvasHeight = 512;

const canvas = document.getElementById("canvas");
const overlay = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const overlayCtx = overlay.getContext("2d");

const rootStyles = getComputedStyle(document.documentElement);


export function setupCanvas() {
    let drawing = false;
    let erasing = false;
    let lastX = 0;
    let lastY = 0;
    let x = 0;
    let y = 0;

    // Setup canvas
    canvas.height = canvasHeight;
    canvas.width = canvasWidth;
    overlay.height = canvas.height;
    overlay.width = canvas.width;

    // Setup event listeners
    canvas.addEventListener("mousedown", (e) => {
        if (e.button === 0) {
            drawing = true;

            const rect = canvas.getBoundingClientRect();
            lastX = (e.clientX - rect.left) * canvas.width / rect.width;
            lastY = (e.clientY - rect.top) * canvas.height / rect.height;
        }
    });
    canvas.addEventListener("mouseup", () => {
        drawing = false;
    });
    canvas.addEventListener("mouseleave", () => {
        drawing = false;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    });
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();

        // Normalize direction: deltaY > 0 means scrolling down
        if (e.deltaY < 0) {
            radiusCurrent = Math.min(radiusCurrent + radiusStep, radiusMax);
        } else {
            radiusCurrent = Math.max(radiusCurrent - radiusStep, radiusMin);
        }

        updateOverlay(x, y);

        console.log("Brush size:", radiusCurrent);
    });
    canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        x = (e.clientX - rect.left) * canvas.width / rect.width;
        y = (e.clientY - rect.top) * canvas.height / rect.height;

        updateOverlay(x, y);
        if (!drawing) return;

        // Normal drawing
        ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
        ctx.strokeStyle = rootStyles.getPropertyValue('--color-primary-hover').trim();
        ctx.lineWidth = radiusCurrent * 2;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.closePath();

        lastX = x;
        lastY = y;
    });

    // Setup tools
    const toolBox = document.getElementById("toolbox");
    const toolBtns = {
        pen: {
            svg: penSvg,
            cb: () => { erasing = false; },
            selectable: true
        },
        eraser: {
            svg: eraserSvg,
            cb: () => { erasing = true; },
            selectable: true
        },
        clear: {
            svg: clearSvg,
            cb: () => { ctx.clearRect(0, 0, canvas.width, canvas.height); },
        }
    };

    let activeButton = null;

    for (const key in toolBtns) {
        const btnData = toolBtns[key];

        const button = document.createElement("button");

        const activateBtn = () => {
            btnData.cb();
            if (btnData.selectable) {
                // Remove 'active' from previous button
                if (activeButton) activeButton.classList.remove('active');

                // Set this button as active
                button.classList.add('active');
                activeButton = button;
            }
        };

        button.className = "tool-button";
        button.addEventListener("click", activateBtn);

        const icon = document.createElement("img");
        icon.src = btnData.svg;
        icon.alt = key;
        icon.className = 'tool-icon';

        button.prepend(icon);
        toolBox.appendChild(button);

        if (key === "pen") {
            activateBtn();
        }
    }

}

export function getCanvasImageData(width, height) {
    // create an offscreen canvas
    const offCanvas = document.createElement("canvas");
    offCanvas.width = width;
    offCanvas.height = height;
    const offCtx = offCanvas.getContext("2d");

    // draw the original canvas scaled into the offscreen canvas
    offCtx.drawImage(canvas, 0, 0, width, height);

    // get the pixel data
    const imageData = offCtx.getImageData(0, 0, width, height);
    return imageData;
}

function updateOverlay(x, y) {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, radiusCurrent, 0, Math.PI * 2);
    overlayCtx.strokeStyle = rootStyles.getPropertyValue('--color-primary-hover').trim();
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
    overlayCtx.closePath();
}
