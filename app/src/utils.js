/*
 * utils.js
 * Copyright (C) 2025 stantonik <stantonik@stantonik-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

import './style.css';

// Build palette object
const buildPalette = () => {
    // Helper to read CSS variables from :root
    const getVar = (name) =>
        getComputedStyle(document.documentElement)
            .getPropertyValue(name)
            .trim();

    const palette = {
        background: {
            primary: getVar('--bg-primary'),
            secondary: getVar('--bg-secondary'),
            accent: getVar('--bg-accent'),
        },
        text: {
            primary: getVar('--text-primary'),
            secondary: getVar('--text-secondary'),
            muted: getVar('--text-muted'),
        },
        border: {
            color: getVar('--border-color'),
        },
        color: {
            primary: getVar('--color-primary'),
            primaryHover: getVar('--color-primary-hover'),
            secondary: getVar('--color-secondary'),
            error: getVar('--color-error'),
        },
    }
    return palette
};

export let PALETTE = buildPalette();

window.addEventListener('load', () => {
    refreshPalette();
});

export const refreshPalette = () => Object.assign(PALETTE, buildPalette());

export function cvtImageToMNISTInput(imageData) {
    const { data, width, height } = imageData;
    const grayArray = new Float32Array(width * height);

    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4 + 0];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        // simple average, normalized to [0,1]
        grayArray[i] = (r + g + b) / (3 * 255) > 0.2 ? 1 : -1;
    }

    return grayArray;
}

export function cvtMNISTInputToImage(arr, width, height) {
    const imageData = new ImageData(width, height);
    for (let i = 0; i < arr.length; i++) {
        const v = Math.round((arr[i] + 1) / 2 * 255); // convert [0,1] → [0,255]
        imageData.data[i * 4 + 0] = v; // R
        imageData.data[i * 4 + 1] = v; // G
        imageData.data[i * 4 + 2] = v; // B
        imageData.data[i * 4 + 3] = 255; // A
    }
    return imageData;
}


export function indexArray(iterable) {
    return Object.fromEntries(
        Array.from(iterable).map((value, index) => [index, value])
    );
}

export function softmax(arr) {
    const max = Math.max(...arr); // for numerical stability
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}

export function makeImageLink(image, width, height) {
    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = width;
    tmpCanvas.height = height;
    const tmpCtx = tmpCanvas.getContext("2d");
    tmpCtx.putImageData(image, 0, 0);
    return tmpCanvas.toDataURL('image/png');
}
