/*
 * vite.config.js
 * Copyright (C) 2025 stantonik <stantonik@stantonik-mba.local>
 *
 * Distributed under terms of the MIT license.
 */


import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

export default defineConfig({
    base: "/mnist/",
    plugins: [viteSingleFile()],
    build: {
        minify: false,
        cssCodeSplit: false,
    },
})

