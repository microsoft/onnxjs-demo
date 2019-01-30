const fs = require("fs");
const path = require("path");

// copy ONNX.js WebAssembly files to {workspace}/public/ folder
const srcFolder = path.join(__dirname, 'node_modules', 'onnxjs', 'dist');
const destFolder = path.join(__dirname, 'public');
fs.copyFileSync(path.join(srcFolder, 'onnx-wasm.wasm'), path.join(destFolder, 'onnx-wasm.wasm'));
fs.copyFileSync(path.join(srcFolder, 'onnx-worker.js'), path.join(destFolder, 'onnx-worker.js'));
