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

    // Drawing callbacks
    // Helper to get canvas coordinates from mouse or touch
    const getPosFromEvent = (e) => {
        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;

        if (e.touches) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }

        const x = (clientX - rect.left) * canvas.width / rect.width;
        const y = (clientY - rect.top) * canvas.height / rect.height;
        return { x, y };
    };

    // Scroll to change radius
    const editRadius = (deltaOrScale) => {
        if (typeof deltaOrScale === "number") {
            // desktop wheel: delta positive or negative
            radiusCurrent = deltaOrScale < 0
                ? Math.min(radiusCurrent + radiusStep, radiusMax)
                : Math.max(radiusCurrent - radiusStep, radiusMin);
        } else if (typeof deltaOrScale === "object") {
            // pinch: { scale: number }
            radiusCurrent = Math.min(radiusMax, Math.max(radiusMin, radiusCurrent * deltaOrScale.scale));
        }

        updateOverlay(x, y);
        console.log("Brush size:", radiusCurrent);
    };

    const handleWheel = (e) => {
        e.preventDefault();
        editRadius(e.deltaY);
    };

    let initialPinchDistance = null;

    const getDistance = (t1, t2) => Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);

    const handleTouchStart = (e) => {
        if (e.touches.length === 2) {
            initialPinchDistance = getDistance(e.touches[0], e.touches[1]);
        }
    };

    const handleTouchMove = (e) => {
        if (e.touches.length === 2 && initialPinchDistance) {
            const currentDistance = getDistance(e.touches[0], e.touches[1]);
            const scale = currentDistance / initialPinchDistance;
            editRadius({ scale });
        }
    };

    const handleTouchEnd = (e) => {
        if (e.touches.length < 2) {
            initialPinchDistance = null;
        }
    };

    // Start drawing
    const startDrawing = (e) => {
        e.preventDefault(); // prevent scrolling on touch
        drawing = true;
        const pos = getPosFromEvent(e);
        lastX = pos.x;
        lastY = pos.y;
    };

    // Stop drawing
    const stopDrawing = (e) => {
        e.preventDefault();
        drawing = false;
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    };

    // Draw on canvas
    const draw = (e) => {
        e.preventDefault();

        if (e.touches && e.touches.length > 1) return;

        const pos = getPosFromEvent(e);
        x = pos.x;
        y = pos.y;

        updateOverlay(x, y);

        if (!drawing) return;

        ctx.globalCompositeOperation = erasing ? 'destination-out' : 'source-over';
        ctx.strokeStyle = rootStyles.getPropertyValue('--color-primary-hover').trim();
        ctx.lineWidth = radiusCurrent * 2;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.closePath();

        lastX = x;
        lastY = y;
    };

    // Setup event listeners
    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('wheel', handleWheel);

    // Touch events
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchcancel', stopDrawing);
    canvas.addEventListener('touchmove', draw, { passive: false });

    // Touch events (pinch)
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd);
    canvas.addEventListener('touchcancel', handleTouchEnd);



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
