import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

/** The editor's unit tests (`tests/`): its pure modules and store, in Node, without a browser. */
export default defineConfig({
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  test: { include: ["tests/**/*.test.ts"], environment: "node" },
});
