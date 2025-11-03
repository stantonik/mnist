/*
 * inference.js
 * Copyright (C) 2025 stantonik <stantonik@stantonik-mba.local>
 *
 * Distributed under terms of the MIT license.
 */

// Yliess HATI code
let net;

const modelUrl = "/models";

const models = ["mnist_mlp", "mnist_convnet"];
const modelSelect = document.getElementById("model");
const statusText = document.getElementById("status");
const timerText = document.getElementById("timer");

const error = (err) => {
    statusText.innerHTML = `Error: ${err}`;
    throw new Error(err);
};

const timer = async (func, label = "") => {
    timerText.innerHTML = "";
    const start = performance.now();
    const out = await func();
    const delta = (performance.now() - start).toFixed(1);
    console.log(`${delta} ms ${label}`);
    timerText.innerHTML = `${delta} ms ${label}`;
    return out;
};

const getDevice = async () => {
    if (!navigator.gpu) error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    return await adapter.requestDevice({
        requiredFeatures: ["shader-f16"],
        powerPreference: "high-performance",
    });
};

const loadNet = async (modelName) => {
    const jsPath = `.${modelUrl}/${modelName}/${modelName}.js`;
    const netPath = `${modelUrl}/${modelName}/${modelName}.webgpu.safetensors`;
    try {
        statusText.innerHTML = "loading the model...";
        const device = await getDevice();
        const tinygrad = (await import(jsPath)).default;
        net = await timer(
            () => tinygrad.load(device, netPath),
            "(compilation)",
        );
        statusText.innerHTML = "ready to classify";
    } catch (e) {
        error(e);
    }
};

export const makePrediction = async (image) => {
    if (!net) error("Net not loaded yet.");
    const res = await timer(
        () => net(new Float32Array(image)),
        "(inference)",
    );
    const logits = Array.from(new Float32Array(res[0]));
    return logits;
};

export const setupInference = async () => {
    for (const model of models) {
        const modelOpt = document.createElement("option");
        modelOpt.value = model;
        modelOpt.innerHTML = model;
        modelSelect.appendChild(modelOpt);
    }
    modelSelect.addEventListener("change", (e) => loadNet(e.target.value));
    await loadNet(modelSelect.value);
};


