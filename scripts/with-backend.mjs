#!/usr/bin/env node
/**
 * Run a Next.js command with the segmentation backend (`python -m mrfseg.server`) beside it,
 * and stop both together. `npm run dev` and `npm start` go through here:
 *
 *     node scripts/with-backend.mjs next dev -H 127.0.0.1 -p 8003
 *
 * Without python/.venv the editor still starts, and its Segmentation panel says the backend
 * is off. Both sides take the backend's port from MRFSEG_BACKEND_PORT (default 8004).
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const WINDOWS = process.platform === "win32";
const PYTHON = path.resolve(WINDOWS ? "python\\.venv\\Scripts\\python.exe" : "python/.venv/bin/python");
/** The backend's exit status when its port is taken (`mrfseg/server/app.py`). */
const EXIT_PORT_IN_USE = 2;
const children = new Set();
let stopping = false;

function stopAll(signal = "SIGTERM") {
  stopping = true;
  for (const child of children) child.kill(signal);
}

function start(name, file, argv, options = {}) {
  const startedAt = Date.now();
  const child = spawn(file, argv, { stdio: "inherit", ...options });
  children.add(child);
  child.on("error", (error) => {
    console.error(`[${name}] could not start: ${error.message}`);
    if (name === "editor") {
      stopAll();
      process.exitCode = 1;
    }
  });
  child.on("exit", (code, signal) => {
    children.delete(child);
    if (name === "editor") {
      // The editor is the point of the command: when it ends, so does the backend.
      stopAll();
      process.exitCode = code ?? (signal ? 1 : 0);
    } else if (!stopping) {
      console.error(`[mrfseg] backend stopped (${signal ?? `exit code ${code}`}); the editor carries on without it.`);
      if (code !== 0 && code !== EXIT_PORT_IN_USE && Date.now() - startedAt < 30_000) {
        console.error("[mrfseg] If it is missing a module: python/.venv/bin/pip install -r python/requirements.txt");
      }
    }
  });
}

/** `next ...` as node running Next's own script: the same on every platform, with no shell. */
function editorCommand([command, ...args]) {
  if (command === "next") return [process.execPath, [createRequire(import.meta.url).resolve("next/dist/bin/next"), ...args]];
  return [command, args];
}

if (existsSync(PYTHON)) {
  start("backend", PYTHON, ["-m", "mrfseg.server"], { cwd: "python" });
} else {
  console.warn("[mrfseg] python/.venv not found, so segmentation is off. See the README: Python core.");
}
start("editor", ...editorCommand(process.argv.slice(2)));

for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => stopAll(signal));
