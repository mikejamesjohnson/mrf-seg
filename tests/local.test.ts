import { describe, expect, it } from "vitest";

import { isLoopbackHost, refusal } from "@/lib/server/local";

const request = (headers: Record<string, string>) => new Request("http://127.0.0.1:8003/api/file?path=/x.mha", { headers });

describe("isLoopbackHost", () => {
  it("takes this computer's names, with or without a port", () => {
    for (const host of ["127.0.0.1", "127.0.0.1:8003", "localhost:8003", "LOCALHOST", "[::1]", "[::1]:8003"]) {
      expect(isLoopbackHost(host), host).toBe(true);
    }
  });

  it("refuses any other, including one that only starts like it", () => {
    for (const host of ["example.com", "127.0.0.1.example.com", "192.168.1.2:8003", "[::2]:8003", ""]) {
      expect(isLoopbackHost(host), host).toBe(false);
    }
    expect(isLoopbackHost(null)).toBe(false);
  });
});

describe("refusal", () => {
  it("serves the editor's own page, and the address bar", () => {
    expect(refusal(request({ host: "127.0.0.1:8003", "sec-fetch-site": "same-origin" }))).toBeNull();
    expect(refusal(request({ host: "127.0.0.1:8003", "sec-fetch-site": "none" }))).toBeNull();
    expect(refusal(request({ host: "127.0.0.1:8003", origin: "http://127.0.0.1:8003" }))).toBeNull();
  });

  it("refuses another site, another host, and an opaque origin", () => {
    expect(refusal(request({ host: "127.0.0.1:8003", "sec-fetch-site": "cross-site" }))).not.toBeNull();
    expect(refusal(request({ host: "evil.example:8003" }))).not.toBeNull();
    expect(refusal(request({ host: "127.0.0.1:8003", origin: "http://evil.example" }))).not.toBeNull();
    expect(refusal(request({ host: "127.0.0.1:8003", origin: "null" }))).not.toBeNull();
  });
});
